# HealthSearch

Protótipo de busca híbrida para documentos médicos, desenvolvido para o desafio integrador de Recuperação de Informação e Processamento de Linguagem Natural.

## Entregáveis

- `healthsearch_app.py`: aplicação Streamlit completa.
- `relatorio_healthsearch.pdf`: relatório técnico do projeto.
- `requirements.txt`: dependências do projeto.

## Funcionalidades

- Pré-processamento em português com normalização, tokenização e stopwords.
- Ranking léxico com Okapi BM25 e sliders para `k1` e `b`.
- Busca semântica com embeddings e similaridade de cosseno.
- Fusão dos rankings com Reciprocal Rank Fusion (RRF), usando `k_RRF = 60`.
- Abas comparativas para BM25, busca semântica, RRF e matriz de rankings.
- Re-ranking opcional dos três primeiros candidatos com Cross-Encoder.
- Corpus médico fixo com seis documentos do enunciado.

## Como executar

Requer Python 3.10 ou superior.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run healthsearch_app.py
```

Na primeira execução, `sentence-transformers` pode baixar os modelos de embedding e Cross-Encoder. O checkbox do Cross-Encoder é opcional e fica desativado por padrão.

## Verificação rápida

```bash
python -m py_compile healthsearch_app.py
```
