import pandas as pd

from pathlib import Path

# Isso pega o diretório onde o script atual está salvo
pasta_do_script = Path(__file__).resolve().parent

# Isso monta o caminho completo, independentemente de onde você rodou o comando
caminho_do_arquivo = pasta_do_script / "openrouter.csv"

# Agora você pode abrir sem erro
with open(caminho_do_arquivo, "r") as f:
    print(f.read())

df = pd.read_csv(caminho_do_arquivo)

# Converte a coluna de data
df['created_at'] = pd.to_datetime(df['created_at'])

# Filtra o horário desejado
filtro = df[df['created_at'] >= '2026-06-26 17:03:00']

# Calcula os totais
print(f"Total Input Tokens: {filtro['tokens_prompt'].sum()}")
print(f"Total Output Tokens: {filtro['tokens_completion'].sum()}")
print(f"Total Gasto (USD): {filtro['cost_total'].sum():.6f}")