# Dashboard Gerencial — Trilhas
 Teste livia
Dashboard estático (HTML + JS) para acompanhamento de **progresso**, **pontos** e **conclusões** por **trilha** e **curso**, com visualização por **mês**, lista de **alunos** e uma seção de **insights automáticos**.

> **Stack**: HTML5, CSS, Chart.js, JavaScript vanilla.  
> **Dados**: arquivo `data_trilhas.json` (pré-processado fora do front).

---

## 🧭 Visão Geral

- **Filtros**: `Trilha` (ou visão geral com todas) e `Mês`; opção de **acumular pontos** até o mês.
- **KPIs** (respeitando os filtros):
  - **Alunos ativos (snapshot ≤ mês)**: quantidade de alunos com algum progresso conhecido até o mês.
  - **Progresso médio (entre quem iniciou)**: média de progresso apenas de quem tem progresso > 0.
  - **Conclusões (no mês / acumulado)**.
  - **Pontos (no mês / acumulado)** — **pontos manuais não são somados no ranking de cursos** (apenas exibidos na aba Alunos).
- **Gráficos**:
  - **Progresso médio — mês selecionado**: barras; visão geral compara trilhas no mês; por trilha mostra o valor da trilha no mês.
  - **Pontos — mês a mês**: linha; alterna entre mês a mês e acumulado.
- **Top cursos — visão geral (acumulado)**: cursos ordenados por progresso médio, com colunas:
  - Curso
  - **Alunos (com acesso)**
  - **Progresso Médio (entre quem iniciou)**
  - **Pontos (não manuais)**
- **Alunos**: tabela dos alunos/cursos no mês filtrado. Inclui uma **linha “Pontos Manuais”** por aluno quando existir pontuação manual no mês (sem curso/trilha).
- **Insights automáticos** (texto):
  - Tendência de progresso (Δ p.p. vs mês anterior, quando houver).
  - **Top 3** e **Bottom 3** cursos por progresso médio no mês.
  - Resumo de **pontos manuais** no mês.
  - Checagem rápida de **consistência** (progresso > 0 e pontos não manuais = 0).

---

## 📁 Estrutura do Repositório
<img width="708" height="282" alt="image" src="https://github.com/user-attachments/assets/b03b394a-e76d-4386-97b7-7d048c469dd8" />


## Atualização pelo VSCODE
cd ~/Dashboard_Conectajl

# 1) ver o que mudou
git status

# 2) adicionar tudo que mudou
git add .

# 3) criar o commit
git commit -m "Atualiza preprocess e dashboard (provas/nota média)"

# 4) enviar pro GitHub (branch atual)
git push

