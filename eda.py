#%%
from IPython.core import getipython
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pygam import ExpectileGAM
from sklearn.linear_model import QuantileRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
# %%

df=pd.read_csv(r'data/raw/sales_data_analysis.csv')
# %%
print(f'Quantidade de colunas - {len(list(df.columns))!s}\n{50*"="}')
print(df.dtypes)
print(f'\n{50*"="}\n{df.count()}')
df.head(10)

#%%
df['Manager'].value_counts()
#%%
df['Manager'] = df['Manager'].str.strip().str.replace(r'\s+', ' ', regex=True) 
df['Manager'].value_counts()

#%%
df.dtypes
df['Quantity']=df['Quantity'].astype(int)
df['Revenue']=df['Price']*df['Quantity']
# %%

df.head(20)

#%%
df[['Product','Price']].value_counts().sort_values()
# Correção da remoção de linhas com preços corrompidos (29.05 e 25.50 estavam na coluna Price)
df.drop(df.loc[(df['Product']=='Chicken Sandwiches') & (df['Price'].isin([29.05, 29.90]))].index, inplace=True)
df.drop(df.loc[(df['Product']=='Fries') & (df['Price'].isin([25.50, 24.50]))].index, inplace=True)

# %%
def add_price_variation(
    df,
    price_col='Price',
    quantity_col='Quantity',
    city_col='City',
    max_variation=0.25, # Variação máxima de 25%
    elasticity=-1.3, # Elasticidade-preço da demanda
    city_multipliers=None, # Multiplicadores de preço por cidade
    cost_rate=0.35, # % alíquota que representará o custo do produto (ou dict por produto)
    outlier_prob=0.05, # Probabilidade de ocorrência de outliers
    random_state=42 # Seed para reprodutibilidade
):
    """
    Simula uma variação realista de preços e demanda de mercado:
    
    1. Variação por Cidade (City): Reflete custo de vida e poder de compra regional.
       Cidades mais caras (London, Paris) praticam preços ligeiramente maiores,
       enquanto cidades como Madri e Lisboa praticam preços menores. Somado ao
       ruído local diário, a variação chega a até +-25% (max_variation).
       
    2. Elasticidade-Preço da Demanda (Lei da Demanda):
       Q = Q_base * (P / P_base) ** elasticity * ruído_demanda
       - Preços mais baixos geram maior volume vendido (mais casos vendidos).
       - Preços mais altos retraem a demanda (menos casos vendidos).
       
    3. Outliers da Vida Real:
       - Preço: liquidações relâmpago profundas (-30% a -45%) ou cobranças de eventos/aeroporto (+30% a +45%).
       - Demanda: pedidos volumosos em lote/catering (bulk) ou rupturas operacionais de estoque (stockouts).
       
    4. Custo Unitário e Margens:
       - Adiciona coluna 'Cost' (Custo Marginal / Custo Unitário) com base na alíquota percentual fixa sobre o preço base dos produtos.
       - Atualiza Receita ('Revenue'), Custo Total ('Total_Cost') e Lucro ('Profit').
    """
    rng = np.random.default_rng(random_state)
    df = df.copy()

    # Preço e quantidade base por produto (mediana)
    base_prices = df.groupby('Product')[price_col].transform('median')
    base_quantities = df.groupby('Product')[quantity_col].transform('median') if quantity_col in df.columns else None

    # Efeito regional por Cidade e por ruído local
    if city_multipliers is not None:
        city_effect = df[city_col].map(city_multipliers).fillna(0.0) if city_col in df.columns else 0.0
        max_city_effect = np.abs(list(city_multipliers.values())).max() if city_multipliers else 0.0
        remaining_var = max(0.05, max_variation - max_city_effect)
        local_noise = rng.normal(0, remaining_var / 2.0, size=len(df))
    else:
        city_effect = 0
        local_noise = rng.normal(0, max_variation / 2.0, size=len(df))
    
    # Limitar o ruído/variação local pelo percentual de variação máximo (default=25%)
    total_pct_var = np.clip(city_effect + local_noise, -max_variation, max_variation)
    
    # Cria outliers de Preço (promoções extremas ou sobrepreço de eventos) e vincula eles a datas comemorativas
    is_price_outlier = rng.random(size=len(df)) < (outlier_prob / 2.0)
    outlier_direction = rng.choice([-1, 1], size=len(df))
    extreme_price_var = outlier_direction * rng.uniform(0.30, 0.45, size=len(df))
    total_pct_var = np.where(is_price_outlier, extreme_price_var, total_pct_var)
    
    # Atualiza coluna de Preço
    df[price_col] = (base_prices * (1 + total_pct_var)).round(2)
    df[price_col] = np.maximum(df[price_col], 0.50) # salvaguarda mínima de valor
    
    # Elasticidade da Demanda (atualização da Quantidade vendida)
    if base_quantities is not None:
        price_ratio = df[price_col] / base_prices
        
        # Suporte a elasticidade específica por produto ou genérica
        if isinstance(elasticity, dict):
            prod_elasticity = df['Product'].map(elasticity).fillna(-1.2)
        else:
            prod_elasticity = elasticity
            
        # Ruído diário na demanda (clima, dia da semana, fluxo de clientes) para balancear o dataset
        daily_demand_noise = rng.normal(1.0, 0.06, size=len(df))
        simulated_qty = base_quantities * (price_ratio ** prod_elasticity) * daily_demand_noise
        
        df[quantity_col] = np.round(np.maximum(simulated_qty, 1)).astype(int)
    
    # 4. Alíquota de custo fixo (% do preço base) representando o Custo Unitário
    if isinstance(cost_rate, dict):
        prod_cost_rate = df['Product'].map(cost_rate).fillna(0.35)
    else:
        prod_cost_rate = cost_rate
    
    df['Cost'] = (base_prices * prod_cost_rate).round(2)

    # 5. Atualização de Receita, Custo Total e Lucro
    if 'Revenue' in df.columns or quantity_col in df.columns:
        df['Revenue'] = (df[price_col] * df[quantity_col]).round(2)
        df['Total_Cost'] = (df['Cost'] * df[quantity_col]).round(2)
        df['Profit'] = (df['Revenue'] - df['Total_Cost']).round(2)
        
    return df

# Aplica variação realista de preços (até 25%), efeito por cidade, elasticidade e outliers

city_multipliers = {
    'London': 0.10,
    'Paris': 0.07,
    'Berlin': 0.00,
    'Madrid': -0.06,
    'Lisbon': -0.09
}

df = add_price_variation(
    df,
    price_col='Price',
    quantity_col='Quantity',
    city_col='City',
    max_variation=0.25,
    elasticity=-1.118,
    city_multipliers=city_multipliers,
    cost_rate=0.18,
    outlier_prob=0.05,
    random_state=42
)

# %%
df[['Product','Price']].loc[df['Product']=='Chicken Sandwiches'].sort_values('Price', ascending=True).tail(10)

#%%

df.loc[(df['Product']=='Fries')].sort_values('Quantity', ascending=False).head(10)

# %%
# Criando o boxplot se baseando na categoria
sns.boxplot(data=df, 
            x='Product', 
            y='Price',
            palette='Set2')

plt.title('Distribuição do Preço por Categoria')
plt.xlabel('Categoria')
plt.ylabel('Preço ($)')
plt.show()

# %%
list(df['Product'].unique())

# %%

# Criando uma List Comprehension para filtrar colunas categoricas
target_cols = [
    col
    for col, dtype in df.dtypes.items()
    if dtype in ['str'] and df[col].nunique() <= 10
]

# Configurar as grades de visualização dos subplots
n_cols = 2
n_rows = int(np.ceil(len(target_cols) / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4.5 * n_rows))
axes = axes.flatten() if len(target_cols) > 1 else [axes]

sns.set_theme(style='whitegrid')

# Para cada coluna alvo, gera um boxplot
for i, col in enumerate(target_cols):
    ax = axes[i]

    # Ordena pelas categorias mais frequentes dentro da coluna
    order = df[col].value_counts().index

    # Cria o gráfico de contagem
    sns.countplot(
        data=df,
        y=col,
        order=order,
        #palette='Blues_r',
        ax=ax,
    )

    ax.set_title(f'Distribuição: {col}', fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel('Contagem')
    ax.set_ylabel('')

    # Adiciona rótulos com a contagem exata no final de cada barra
    for container in ax.containers:
        ax.bar_label(container, padding=3, fontsize=10)

# Esconde eixos sobrando caso o número de colunas seja ímpar
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

# %%
# Separando treino e teste evitando data leakage
df_train, df_test = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df['Product']
)

print(f"Total de registros: {len(df)}")
print(f"Conjunto de Treino (80%): {len(df_train)} amostras")
print(f"Conjunto de Teste (20%): {len(df_test)} amostras")

# %%
# 2. Remoção de outliers por IQR calculada no treino
iqr_bounds = {}
for prod in df_train['Product'].unique():
    prod_prices = df_train.loc[df_train['Product'] == prod, 'Price']
    q1 = prod_prices.quantile(0.25)
    q3 = prod_prices.quantile(0.75)
    iqr = q3 - q1
    iqr_bounds[prod] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)

# Aplica o filtro exclusivamente no conjunto de treino
df_train_clean = df_train[
    df_train.apply(lambda row: iqr_bounds[row['Product']][0] <= row['Price'] <= iqr_bounds[row['Product']][1], axis=1)
].copy()

print(f"Treino antes do filtro de IQR: {len(df_train)} amostras")
print(f"Treino após filtro de IQR: {len(df_train_clean)} amostras ({len(df_train) - len(df_train_clean)} outliers removidos)")

# %%
def model_predict_quantiles(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame = None,
    method: str = 'Log-log',
    quantiles: list = [0.1, 0.25, 0.5, 0.75, 0.9]
):
    """
    Treina modelos de quantis (Log-log ou GAM) estritamente no conjunto de treino para cada produto.
    """
    unique_products = list(train_df['Product'].unique())
    calculed_train_quantiles = pd.DataFrame()
    calculed_test_quantiles = pd.DataFrame()

    for prod in unique_products:
        sub_train = train_df.loc[train_df['Product'] == prod].copy()

        # Define qual método será utilizado
        if method == 'Log-log':
            X_train = np.log(sub_train[['Price']])
            Y_train = np.log(sub_train['Quantity'])
        elif method == 'GAM':
            X_train = sub_train[['Price']]
            Y_train = sub_train['Quantity']
        else:
            raise ValueError(f"Método não encontrado: {method}, escolha entre 'GAM' ou 'Log-log'.")

        # Prepara o conjunto de testes 
        if test_df is not None:
            sub_test = test_df.loc[test_df['Product'] == prod].copy()
            if method == 'Log-log':
                X_test = np.log(sub_test[['Price']])
            elif method == 'GAM':
                X_test = sub_test[['Price']]

        # Calcula os resultados por Quartil/Expectil
        prod_train_results = {}
        prod_test_results = {}

        for q in quantiles:
            if method == 'Log-log':
                model = QuantileRegressor(quantile=q, alpha=0)
                model.fit(X_train, Y_train)
                prod_train_results[f"q_{q}"] = np.exp(model.predict(X_train))
                if test_df is not None:
                    prod_test_results[f"q_{q}"] = np.exp(model.predict(X_test))
            elif method == 'GAM':
                model = ExpectileGAM(terms='auto', expectile=q)
                model.fit(
                    X_train.values if hasattr(X_train, 'values') else X_train,
                    Y_train.values if hasattr(Y_train, 'values') else Y_train
                )
                prod_train_results[f"q_{q}"] = model.predict(
                    X_train.values if hasattr(X_train, 'values') else X_train
                )
                if test_df is not None:
                    prod_test_results[f"q_{q}"] = model.predict(
                        X_test.values if hasattr(X_test, 'values') else X_test
                    )

        # resultados do treino
        base_cols = [c for c in ['Product', 'Price', 'Cost', 'Quantity', 'Revenue', 'Profit'] if c in sub_train.columns]
        quantiles_train_df = pd.DataFrame(prod_train_results, index=sub_train.index)
        temp_train = pd.concat([sub_train[base_cols], quantiles_train_df], axis=1)
        calculed_train_quantiles = pd.concat([calculed_train_quantiles, temp_train], axis=0)

        # Contecatenando os resultados do teste 
        if test_df is not None:
            base_cols_test = [c for c in ['Product', 'Price', 'Cost', 'Quantity', 'Revenue', 'Profit'] if c in sub_test.columns]
            quantiles_test_df = pd.DataFrame(prod_test_results, index=sub_test.index)
            temp_test = pd.concat([sub_test[base_cols_test], quantiles_test_df], axis=1)
            calculed_test_quantiles = pd.concat([calculed_test_quantiles, temp_test], axis=0)

    # Calculando a receita estimada, lucro e Markup de Lerner
    for q in quantiles:
        calculed_train_quantiles[f'revenue_{q}'] = calculed_train_quantiles[f'q_{q}'] * calculed_train_quantiles['Price']
        if 'Cost' in calculed_train_quantiles.columns:
            calculed_train_quantiles[f'profit_{q}'] = (
                (calculed_train_quantiles['Price'] - calculed_train_quantiles['Cost']) * calculed_train_quantiles[f'q_{q}']
            )

        if test_df is not None:
            calculed_test_quantiles[f'revenue_{q}'] = calculed_test_quantiles[f'q_{q}'] * calculed_test_quantiles['Price']
            if 'Cost' in calculed_test_quantiles.columns:
                calculed_test_quantiles[f'profit_{q}'] = (
                    (calculed_test_quantiles['Price'] - calculed_test_quantiles['Cost']) * calculed_test_quantiles[f'q_{q}']
                )

    # Índice de Lerner observado por linha: L = (P - MC) / P
    if 'Cost' in calculed_train_quantiles.columns:
        calculed_train_quantiles['lerner_index'] = (
            (calculed_train_quantiles['Price'] - calculed_train_quantiles['Cost']) / calculed_train_quantiles['Price']
        )
        if test_df is not None:
            calculed_test_quantiles['lerner_index'] = (
                (calculed_test_quantiles['Price'] - calculed_test_quantiles['Cost']) / calculed_test_quantiles['Price']
            )

    if test_df is not None:
        return calculed_train_quantiles, calculed_test_quantiles
    return calculed_train_quantiles

# %%
quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]

#Log log
df_train_loglog, df_test_loglog = model_predict_quantiles(
    train_df=df_train_clean,
    test_df=df_test,
    method='Log-log',
    quantiles=quantiles
)

# GAM
df_train_gam, df_test_gam = model_predict_quantiles(
    train_df=df_train_clean,
    test_df=df_test,
    method='GAM',
    quantiles=quantiles
)

# %%
# Métricas de validação
def calculate_validation_metrics(df_test_preds: pd.DataFrame, model_name: str = 'Model') -> pd.DataFrame:
    """
    Calcula as métricas de validação importadas do scikit-learn (MAE, MSE, RMSE e R²)
    comparando a demanda real observada no teste ('Quantity') com a predição central ('q_0.5').
    """
    metrics = []

    # Métrica Global (Todos os produtos combinados no teste)
    y_true_global = df_test_preds['Quantity']
    y_pred_global = df_test_preds['q_0.5']

    metrics.append({
        'Modelo': model_name,
        'Produto': 'Global (Todos)',
        'Amostras': len(y_true_global),
        'MAE': mean_absolute_error(y_true_global, y_pred_global),
        'MSE': mean_squared_error(y_true_global, y_pred_global),
        'RMSE': root_mean_squared_error(y_true_global, y_pred_global),
        'R2': r2_score(y_true_global, y_pred_global)
    })

    # Métricas individualmente por produto
    for prod in df_test_preds['Product'].unique():
        sub = df_test_preds[df_test_preds['Product'] == prod]
        y_true = sub['Quantity']
        y_pred = sub['q_0.5']
        metrics.append({
            'Modelo': model_name,
            'Produto': prod,
            'Amostras': len(sub),
            'MAE': mean_absolute_error(y_true, y_pred),
            'MSE': mean_squared_error(y_true, y_pred),
            'RMSE': root_mean_squared_error(y_true, y_pred),
            'R2': r2_score(y_true, y_pred)
        })

    return pd.DataFrame(metrics)

# Avaliação dos modelos no conjunto de teste não visto
metrics_loglog = calculate_validation_metrics(df_test_loglog, model_name='Log-log')
metrics_gam = calculate_validation_metrics(df_test_gam, model_name='GAM')

# Tabela comparativa consolidada
df_validation_metrics = pd.concat([metrics_loglog, metrics_gam], ignore_index=True)

print("=" * 78)
print("RELATÓRIO DE MÉTRICAS DE VALIDAÇÃO NO CONJUNTO DE TESTE")
print("=" * 78)
print(df_validation_metrics.round(4).to_string(index=False))
print("=" * 78)

# %%
# Determinação do Preço Ótimo via Markup de Lerner (Maximização de Lucro / Margem de Contribuição)
# Fórmula: Lucro = (Preço - Custo Marginal) * Quantidade_0.5
def optimization_price_algorithm(df_train: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula a precificação ótima teórica segundo a Regra do Markup de Lerner:
    (P - MC) / P = 1 / |elasticidade| => P* = MC * (|elasticidade| / (|elasticidade| - 1))
    """
    records = []
    for prod in df_train['Product'].unique():
        sub = df_train[df_train['Product'] == prod]

        # Transformam em log-log e aplica regressão quantílica
        X = np.log(sub[['Price']])
        y = np.log(sub['Quantity'])
        qr = QuantileRegressor(quantile=0.5, alpha=0).fit(X, y)
        
        # Calcula elasticidade, que é a diferença da quantidade demandada em função da variação percentual de preço
        elasticity = qr.coef_[0]
        cost = sub['Cost'].iloc[0]
        abs_e = abs(elasticity) #Transforma em absoluto
        
        # Fórmula de Lerner só aceita elasticidade absoluta maior que 1
        if abs_e > 1:
            lerner_markup_factor = abs_e / (abs_e - 1)
            p_opt_lerner = cost * lerner_markup_factor
            lerner_index = 1.0 / abs_e
        else:
            lerner_markup_factor = np.nan
            p_opt_lerner = np.nan
            lerner_index = np.nan
            
        records.append({
            'Product': prod,
            'Cost (MC)': cost,
            'Elasticidade (e)': round(elasticity, 4),
            'Markup Teórico (P/MC)': round(lerner_markup_factor, 4) if pd.notna(lerner_markup_factor) else np.nan,
            'Índice Lerner Teórico (1/|e|)': round(lerner_index, 4) if pd.notna(lerner_index) else np.nan,
            'Preço Ótimo Teórico (Lerner)': round(p_opt_lerner, 2) if pd.notna(p_opt_lerner) else np.nan
        })
    return pd.DataFrame(records)

df_lerner_theoretical = optimization_price_algorithm(df_train_clean)
print("\n" + "=" * 78)
print("REGRA TEÓRICA DO MARKUP DE LERNER POR PRODUTO")
print("=" * 78)
print(df_lerner_theoretical.to_string(index=False))
print("=" * 78)

# 2. Preço Ótimo Empírico via Modelos (GAM e Log-log) maximizando Lucro (profit_0.5)
price_optimo_gam = (
    df_train_gam.sort_values('profit_0.5', ascending=False)
    .drop_duplicates(subset='Product', keep='first')
    .reset_index(drop=True)
)
price_optimo_loglog = (
    df_train_loglog.sort_values('profit_0.5', ascending=False)
    .drop_duplicates(subset='Product', keep='first')
    .reset_index(drop=True)
)

cols_display = ['Product', 'Price', 'Cost', 'lerner_index', 'Quantity', 'revenue_0.5', 'profit_0.5']

print('\nPreço Ótimo (Markup de Lerner / Lucro) - GAM:')
print(price_optimo_gam[cols_display].round(4).to_string(index=False))

print('\nPreço Ótimo (Markup de Lerner / Lucro) - Log-log:')
print(price_optimo_loglog[cols_display].round(4).to_string(index=False))


# %%

def visualize_predictions(df_input: pd.DataFrame, price_optimo: pd.DataFrame, method: str):
    sns.set_theme(style="whitegrid")

    # Inicializar o grid facetado por produto
    g = sns.FacetGrid(
        data=df_input,
        col="Product",
        col_wrap=2,
        sharex=False,
        sharey=False,
        height=4,
        aspect=1.2,
    )

    # Função para desenhar as camadas por produto
    def _plot_product_facet(data, **kws):
        # Ordenar por 'Price'
        df_sorted = data.sort_values("Price")

        # Intervalo dos quantis 0.1 a 0.9
        plt.fill_between(
            df_sorted["Price"],
            df_sorted["q_0.1"],
            df_sorted["q_0.9"],
            color="#d3d3d3",
            alpha=0.75,
            label="Intervalo (q0.1 - q0.9)",
        )

        # Predição central da mediana (q=0.5)
        plt.plot(
            df_sorted["Price"],
            df_sorted["q_0.5"],
            color="blue",
            linewidth=1.3,
            label="Predição (q=0.5)",
        )

        # Dispersão dos dados observados
        plt.scatter(
            data["Price"],
            data["Quantity"],
            alpha=0.5,
            s=25,
            color="#2b5c8f",
            label="Observado",
        )

        # Destaque em vermelho do preço ótimo do produto
        product_name = data["Product"].iloc[0]
        opt = price_optimo[price_optimo["Product"] == product_name]

        if not opt.empty:
            plt.scatter(
                opt["Price"],
                opt["q_0.5"],
                color="red",
                s=80,
                edgecolors="black",
                linewidths=1.2,
                zorder=5,
                label="Preço Ótimo (Markup de Lerner)",
            )

    # Mapear os dados na função para cada faceta
    g.map_dataframe(_plot_product_facet)

    # Ajustes de rótulos, títulos, legenda e layout
    g.set_axis_labels("Price", "Quantity Sold")
    g.set_titles(col_template="{col_name}", size=12, weight="bold")
    g.add_legend()
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle(f"{method} - Otimização de Preço via Markup de Lerner", fontsize=14, fontweight="bold")

    plt.show()

# %%
# Visualização do ajuste e preço ótimo com dados de Treino
visualize_predictions(df_input=df_train_gam, price_optimo=price_optimo_gam, method='GAM (Treino)')

# %% 
visualize_predictions(df_input=df_train_loglog, price_optimo=price_optimo_loglog, method='Log-log (Treino)')

# %%
# Visualização das predições contra os dados observados no conjunto de Teste (Validação)
visualize_predictions(df_input=df_test_gam, price_optimo=price_optimo_gam, method='GAM (Teste/Validação)')

# %%
visualize_predictions(df_input=df_test_loglog, price_optimo=price_optimo_loglog, method='Log-log (Teste/Validação)')




# %%
