#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pygam import ExpectileGAM, LinearGAM,s
from sklearn.preprocessing import OneHotEncoder
# %%

df=pd.read_csv('sales_data_analysis.csv')
# %%
print(f'Quantidade de colunas - {str(len(list(df.columns)))}\n{50*"="}')
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
df.rename(
    columns={
        'Purchase Type': 'purchase_type',
    },
    inplace=True,
)

# %%
encoder = OneHotEncoder(sparse_output=False)
dados_codificados = encoder.fit_transform(df[['purchase_type']])
nomes_colunas = encoder.get_feature_names_out(['purchase_type'])
df_codificado = pd.DataFrame(dados_codificados, columns=nomes_colunas)
pd.concat([df, df_codificado], axis=1)

# %%



# Calcula o Expectile (Quartil com GAM) iterando por Produto
unique_products=list(df['Product'].unique())
calculed_expectiles=pd.DataFrame()

for i in unique_products:
    df_product = df.loc[df['Product'] == i]
    quantiles=[0.1,0.25,0.5,0.75,0.9]
    gam_results={}
    X=df_product[['Price','purchase_type_Drive-thru','purchase_type_In-store','purchase_type_Online']]
    Y=df_product['Quantity']

    for q in quantiles:
        gam= ExpectileGAM(terms='auto',expectile=q)
        gam.fit(X, Y)
        gam_results[f"q_{q}"] = gam.predict(X)
    
    gam_calcuted=pd.DataFrame(gam_results).set_index(X.index)
    temp_df=pd.concat([df_product[['Product','Price','Quantity','Revenue']],gam_calcuted],axis=1)
    
    calculed_expectiles=pd.concat([calculed_expectiles,temp_df],axis=0)

calculed_expectiles


# %%
for q in quantiles:
    calculed_expectiles[f'revenue_{q}'] = calculed_expectiles[f'q_{q}'] * calculed_expectiles['Price']

# %%
price_optimo=calculed_expectiles.sort_values('revenue_0.5',ascending=False).drop_duplicates(subset='Product',keep='first').reset_index(drop=True)

price_optimo.head(10)
# %%

sns.set_theme(style="whitegrid")

# 2. Inicializar o grid facetado por produto com escalas livres
g = sns.FacetGrid(
    data=calculed_expectiles,
    col="Product",
    col_wrap=2,  # Ajuste o número de colunas conforme preferir
    sharex=False,  # scales="free"
    sharey=False,
    height=4,
    aspect=1.2,
)

# 3. Função para desenhar as camadas de cada faceta
def plot_gam_layer(data, **kws):
    # É fundamental ordenar por 'price' para o ribbon e a linha ficarem contínuos
    df_sorted = data.sort_values("Price")

    # geom_ribbon (Intervalo 2.5% a 97.5%)
    plt.fill_between(
        df_sorted["Price"],
        df_sorted["q_0.1"],
        df_sorted["q_0.9"],
        color="#d3d3d3",
        alpha=0.75,
    )

    # geom_line (Predição central)
    plt.plot(
        df_sorted["Price"],
        df_sorted["q_0.5"],
        color="blue",
        linewidth=1.3,
        label="Predição",
    )

    # geom_point (Dispersão dos dados observados)
    plt.scatter(
        data["Price"],
        data["Quantity"],
        alpha=0.5,
        s=25,
        color="#2b5c8f",  # Ajuste para a cor desejada dos pontos
    )
    # Destaque em vermelho do preço ótimo calculado para o revenue do quartil 0.5 (q_0.5)
    product_name = data["Product"].iloc[0]
    opt = price_optimo[price_optimo["Product"] == product_name]

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


# 4. Mapear os dados na função
g.map_dataframe(plot_gam_layer)

# 5. Ajustes de rótulos, títulos e tamanho
g.set_axis_labels("Price", "Quantity Sold")
g.set_titles(col_template="{col_name}")
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle("GAM Price vs Quantity Model", fontsize=14, fontweight="bold")

plt.show()
# %%
# Comparação utilzando modelo log-log com o PyGAM

gam_loglog_results=pd.DataFrame()
for i in unique_products:
    df_product = df.loc[df['Product'] == i]
    df_product['ln_price'] = np.log(df_product['Price'])
    df_product['ln_quantity'] = np.log(df_product['Quantity'])

    X=df_product['ln_price']
    Y=df_product['ln_quantity']
    gam_loglog = LinearGAM(
        s(0, n_splines=15),
        fit_intercept=True
    )

    # 3. Treinar o modelo
    gam_loglog.fit(X, Y)
    pred_quantities = gam_loglog.predict(X)
    
    gam_loglog_calcuted=pd.DataFrame({'ln_quantities':pred_quantities}).set_index(X.index)
    gam_loglog_calcuted['Quantity_predict']=np.exp(gam_loglog_calcuted['ln_quantities'])
    gam_loglog_calcuted['Revenue_predict'] = gam_loglog_calcuted['Quantity_predict'] * df_product['Price']
    temp_df=pd.concat([df_product[['Product','Price','Quantity','Revenue']],gam_loglog_calcuted],axis=1)

    gam_loglog_results=pd.concat([gam_loglog_results,temp_df],axis=0)

gam_loglog_results
# %%
price_optimo_loglog=gam_loglog_results.sort_values('Revenue_predict',ascending=False).drop_duplicates(subset='Product',keep='first').reset_index(drop=True)

price_optimo_loglog.head(10)
# %%
# %%
