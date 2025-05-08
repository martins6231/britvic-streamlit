# 📊 Dashboard de Acompanhamento de Produção - Britvic

Este projeto é uma aplicação interativa desenvolvida em Python com Streamlit, voltada para o acompanhamento diário da produção de uma indústria de sucos. A solução permite a visualização intuitiva de dados históricos, geração automática de insights e previsão de produção futura com base em séries temporais.

## ✨ Funcionalidades

- Upload de planilhas `.xlsx` com validação automática.
- Painel lateral para seleção de categoria, ano e mês.
- KPIs anuais e resumo estatístico por categoria.
- Gráficos interativos com Plotly:
  - Tendência de produção ao longo do tempo.
  - Variação percentual mensal.
  - Análise de sazonalidade por mês/ano.
  - Comparativos entre anos.
  - Produção acumulada mês a mês.
- Previsão de produção com algoritmo Prophet.
- Detecção de padrões e outliers com geração automática de insights.
- Exportação de dados consolidados com previsões em Excel.

## 📌 Requisitos da Planilha

A planilha deve conter pelo menos as seguintes colunas:

categoria: Nome da categoria de produto.

data: Data da produção (formato reconhecido como data).

caixas_produzidas: Quantidade de caixas produzidas (valores numéricos).

