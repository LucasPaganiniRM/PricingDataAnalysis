#%%
from IPython.core import getipython
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pygam import ExpectileGAM
from sklearn.linear_model import QuantileRegressor

# %%

df=pd.read_csv('sales_data_analysis.csv')
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
       
    4. Atualização de Receita: df['Revenue'] = df['Price'] * df['Quantity'].
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
    
    # 4. Atualização da Receita
    if 'Revenue' in df.columns or quantity_col in df.columns:
        df['Revenue'] = (df[price_col] * df[quantity_col]).round(2)
        
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
    elasticity=-1.3,
    city_multipliers=city_multipliers,
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
# Remove outliers de acordo com o preço por categoria
# Calcula Q1, Q3 e IQR por produto sem alterar o tamanho do DataFrame
q1 = df.groupby('Product')['Price'].transform(lambda x: x.quantile(0.25))
q3 = df.groupby('Product')['Price'].transform(lambda x: x.quantile(0.75))
iqr = q3 - q1

# Mantém apenas o que está dentro do intervalo aceitável
df = df[(df['Price'] >= (q1 - 1.5 * iqr)) & (df['Price'] <= (q3 + 1.5 * iqr))]

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
# Função dinâmica que calcula utilizando GAM ou Log-Log e retorna dataframe
def model_predict_quantiles(df_input: pd.DataFrame, method: str = 'Log-log',quantiles: list=[0.1, 0.25, 0.5, 0.75, 0.9]) -> pd.DataFrame:
    unique_products = list(df_input['Product'].unique())
    calculed_quantiles = pd.DataFrame()

    for prod in unique_products:
        # 1. Filtra o subset do produto preservando o índice original
        sub_df = df_input.loc[df_input['Product'] == prod].copy()
        
        # 2. Prepara X e Y diretamente do subset
        if method == 'Log-log':
            X = np.log(sub_df[['Price']])
            Y = np.log(sub_df['Quantity'])
        elif method == 'GAM':
            X = sub_df['Price']
            Y = sub_df['Quantity']
        else:
            raise ValueError(f"Método não encontrado: {method}, escolha entre 'GAM' ou 'Log-log'.")

        # 3. Acumula todos os quantis deste produto
        prod_results = {}
        for q in quantiles:
            if method == 'Log-log':
                model = QuantileRegressor(quantile=q, alpha=1.0)
                model.fit(X, Y)
                prod_results[f"q_{q}"] = np.exp(model.predict(X))
            elif method == 'GAM':
                model = ExpectileGAM(terms='auto', expectile=q)
                model.fit(X, Y)
                prod_results[f"q_{q}"] = model.predict(X)

        # 4. Cria o DataFrame com o mesmo índice do subset
        quantiles_df = pd.DataFrame(prod_results, index=sub_df.index)
        
        # 5. Concatena colunas do produto com as predições
        temp_df = pd.concat([sub_df[['Product', 'Price', 'Quantity', 'Revenue']], quantiles_df], axis=1)
        calculed_quantiles = pd.concat([calculed_quantiles, temp_df], axis=0)

    return calculed_quantiles

# %%
quantiles=[0.1, 0.25, 0.5, 0.75, 0.9]
df_loglog=model_predict_quantiles(df_input=df, method='Log-log',quantiles=quantiles)
df_gam=model_predict_quantiles(df_input=df, method='GAM',quantiles=quantiles)

# %%
for q in quantiles:
    df_gam[f'revenue_{q}'] = df_gam[f'q_{q}'] * df_gam['Price']
    df_loglog[f'revenue_{q}'] = df_loglog[f'q_{q}'] * df_loglog['Price']


# %%
price_optimo_gam=df_gam.sort_values('revenue_0.5',ascending=False).drop_duplicates(subset='Product',keep='first').reset_index(drop=True)
price_optimo_loglog=df_loglog.sort_values('revenue_0.5',ascending=False).drop_duplicates(subset='Product',keep='first').reset_index(drop=True)

price_optimo_gam.head(10)
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
                label="Preço Ótimo (q=0.5)",
            )

    # Mapear os dados na função para cada faceta
    g.map_dataframe(_plot_product_facet)

    # Ajustes de rótulos, títulos, legenda e layout
    g.set_axis_labels("Price", "Quantity Sold")
    g.set_titles(col_template="{col_name}", size=12, weight="bold")
    g.add_legend()
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle(f"{method} Predict vs Quantity Model", fontsize=14, fontweight="bold")

    plt.show()

# %%
visualize_predictions(df_input=df_gam, price_optimo=price_optimo_gam,method='GAM')

#%% 
visualize_predictions(df_input=df_loglog, price_optimo=price_optimo_loglog,method='Log-log')

# %%
# Comparação utilzando modelo log-log com o PyGAM

loglog_results=pd.DataFrame()
for i in unique_products:
    df_product = df.loc[df['Product'] == i]
    df_product['ln_price'] = np.log(df_product['Price'])
    df_product['ln_quantity'] = np.log(df_product['Quantity'])

    X=df_product[['ln_price']]
    Y=df_product['ln_quantity']
    results={}

    for q in quantiles:
        model = QuantileRegressor(quantile=q, alpha=1.0)
        model.fit(X,Y)
        pred_quantities = model.predict(X)
        results[f'q_{q}']=np.exp(pred_quantities)
        loglog_calcuted=pd.DataFrame(results).set_index(X.index)

    loglog_calcuted['Quantity_predict']=loglog_calcuted['q_0.5']
    loglog_calcuted['Revenue_predict'] = loglog_calcuted['Quantity_predict'] * df_product['Price']
    temp_df=pd.concat([df_product[['Product','Price','Quantity','Revenue']],loglog_calcuted],axis=1)

    loglog_results=pd.concat([loglog_results,temp_df],axis=0)

loglog_results
# %%

price_optimo=loglog_results.sort_values('revenue_0.5',ascending=False).drop_duplicates(subset='Product',keep='first').reset_index(drop=True)
price_optimo=price_optimo[['Product','Price','Revenue']]
# %%
