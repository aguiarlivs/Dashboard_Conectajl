# preprocess.py — gera data_trilhas.json a partir de progresso.xlsx e pontos.xlsx
# Regras aplicadas:
# - "manual" identificado exclusivamente pela coluna TIPO (case-insensitive)
# - Progresso da trilha no mês = média das médias de progresso por curso (snapshot ≤ mês)
# - Sem rateio de pontos entre cursos
# - Linha "Pontos Manuais" apenas no Top cursos — visão geral (overall)
# - Valores de pontos sempre inteiros (arredondados)
# - NOTA:
#     • SEM lista manual de cursos sem prova
#     • prioridade: produtos_provas.csv define se o curso tem prova (possui_prova)
#     • se o curso não estiver no CSV, usa heurística pelas notas (tem nota => tem prova)
#     • cursos sem prova → não entram em nota média / taxa de aprovação
#     • cursos com prova mas aluno sem nota → "Não realizado"

import pandas as pd
import numpy as np
import json, re, math
from datetime import datetime, timezone


# Corte global para aprovação em prova
NOTA_APROVACAO = 70.0  # ajuste se quiser outro corte (ex: 80.0)


def norm_text(s: str) -> str:
    s = "" if s is None else str(s)
    s = re.sub(r"\s+", " ", s.strip())
    return s


def curso_norm(s: str) -> str:
    """Normaliza nome de curso para comparação (case-insensitive, sem espaços extras)."""
    if s is None:
        return ""
    return re.sub(r"\s+", " ", str(s).strip()).lower()


def norm_key(s):
    return norm_text(s).lower()


def find_col(df: pd.DataFrame, candidates):
    cols = {norm_key(c): c for c in df.columns}
    for cand in candidates:
        got = cols.get(norm_key(cand))
        if got:
            return got
    return None


def ym_from_any(d):
    if d is None or (isinstance(d, float) and np.isnan(d)):
        return None
    try:
        if isinstance(d, (int, float)) and not pd.isna(d):
            dt = pd.to_datetime(d, origin="1899-12-30", unit="D", errors="coerce")
        else:
            dt = pd.to_datetime(d, dayfirst=True, errors="coerce")
    except Exception:
        dt = pd.NaT
    if pd.isna(dt):
        return None
    return f"{dt.year:04d}-{dt.month:02d}"


def fnum(x, default=0.0):
    try:
        if isinstance(x, str):
            x = x.replace("%", "").replace(",", ".")
        v = float(x)
        return default if np.isnan(v) else v
    except Exception:
        return default


# ================== 1) Ler produtos_provas.csv (mapa de cursos com/sem prova) ==================
PRODUTOS_PROVA_MAP = {}  # chave: curso_norm, valor: True (tem prova) / False (sem prova)

try:
    produtos_df = pd.read_csv("produtos_provas.csv")
    produtos_df.columns = [norm_text(c) for c in produtos_df.columns]

    col_curso_csv = find_col(produtos_df, ["nome_produto", "curso", "produto", "nome do produto"])
    col_flag_csv  = find_col(produtos_df, ["possui_prova", "tem_prova", "tem prova", "possui prova"])

    if not col_curso_csv or not col_flag_csv:
        print("⚠️ 'produtos_provas.csv' não possui colunas de curso + possui_prova reconhecíveis.")
        print("   Esperado algo como 'nome_produto' + 'possui_prova'.")
    else:
        def parse_flag(v):
            s = str(v).strip().lower()
            if s in ("sim", "s", "yes", "y", "1", "true", "verdadeiro"):
                return True
            if s in ("nao", "não", "n", "no", "0", "false", "falso", ""):
                return False
            # qualquer coisa estranha tratamos como False (sem prova)
            return False

        tmp = produtos_df[[col_curso_csv, col_flag_csv]].copy()
        tmp["__curso_norm"] = tmp[col_curso_csv].apply(curso_norm)
        tmp["__has_prova"] = tmp[col_flag_csv].apply(parse_flag)
        tmp = tmp[tmp["__curso_norm"] != ""]

        PRODUTOS_PROVA_MAP = (
            tmp.drop_duplicates("__curso_norm", keep="last")
               .set_index("__curso_norm")["__has_prova"]
               .to_dict()
        )
        print(f"ℹ️ produtos_provas.csv carregado: {len(PRODUTOS_PROVA_MAP)} cursos mapeados (com/sem prova).")

except FileNotFoundError:
    print("ℹ️ produtos_provas.csv não encontrado; seguiremos apenas com heurística de notas.")


# ================== 2) Ler planilhas de PROGRESSO e PONTOS ==================
progresso = pd.read_excel("progresso.xlsx")
pontos    = pd.read_excel("pontos.xlsx")
progresso.columns = [norm_text(c) for c in progresso.columns]
pontos.columns    = [norm_text(c) for c in pontos.columns]

# variações de colunas
P_ALUNO = find_col(progresso, ["aluno","nome do aluno","usuário","usuario","nome"])
P_CURSO = find_col(progresso, ["curso","produto","formação","formacao","nome do curso"])
P_TRILHA= find_col(progresso, ["trilha"])
P_PROG  = find_col(progresso, ["progresso (%)","progresso","progress"])
P_DT    = find_col(progresso, [
    "atualizado em","última atualização","ultima atualizacao","updated at",
    "data","data de atualização","data atualizacao","ultimo acesso","último acesso"
])
P_NOTA = find_col(progresso, [
    "nota da prova", "nota prova", "nota (%)",
    "resultado da prova", "resultado"
])

T_ALUNO = find_col(pontos, ["aluno","nome","usuário","usuario"])
T_TIPO  = find_col(pontos, ["tipo","evento"])
T_PONTOS= find_col(pontos, ["pontos"])
T_DATA  = find_col(pontos, ["criado em","data","created at"])
T_CURSO = find_col(pontos, ["curso","produto","formação","formacao"])
T_TRILHA= find_col(pontos, ["trilha"])


# ================== 3) Canonicalizar linhas ==================
def map_progresso(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        aluno  = r.get(P_ALUNO) if P_ALUNO else None
        curso  = r.get(P_CURSO) if P_CURSO else None
        trilha = r.get(P_TRILHA) if P_TRILHA else None
        prog   = r.get(P_PROG)  if P_PROG  else None
        dt     = r.get(P_DT)    if P_DT    else None
        nota   = r.get(P_NOTA)  if P_NOTA  else None

        if not aluno or not curso:
            continue

        rows.append({
            "aluno": norm_text(aluno),
            "curso": norm_text(curso),
            "trilha": norm_text(trilha) if trilha and str(trilha).strip() else "(sem trilha)",
            "progresso": fnum(prog, 0.0),
            "nota": fnum(nota, default=np.nan) if nota not in (None, "") else np.nan,
            "atualizado_em": dt,
            "ym": ym_from_any(dt)
        })
    return pd.DataFrame(rows)


def map_pontos(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        aluno = r.get(T_ALUNO) if T_ALUNO else None
        tipo  = r.get(T_TIPO)  if T_TIPO  else ""
        pts   = r.get(T_PONTOS)if T_PONTOS else 0
        dt    = r.get(T_DATA)  if T_DATA  else None
        curso = r.get(T_CURSO) if T_CURSO else ""
        trilha= r.get(T_TRILHA)if T_TRILHA else None
        if not aluno:
            continue

        tipo_s = norm_text(tipo)
        is_manual = (tipo_s.lower() == "manual")  # detecção via TIPO

        rows.append({
            "aluno": norm_text(aluno),
            "tipo":  tipo_s,
            "pontos": fnum(pts, 0.0),
            "criado_em": dt,
            "curso": norm_text(curso),  # pode ser ""
            "trilha_origem": norm_text(trilha) if trilha and str(trilha).strip() else None,
            "is_manual": is_manual,
            "ym": ym_from_any(dt)
        })
    out = pd.DataFrame(rows)
    return out[out["ym"].notna()] if len(out) else out


P = map_progresso(progresso)
T = map_pontos(pontos)


# ================== 4) Cursos com prova / sem prova (sempre priorizando CSV) ==================
CURSO_HAS_PROVA_USE = {}  # chave: curso_norm, valor: True/False

if len(P):
    if "nota" in P.columns:
        cursos_com_nota = set(P.loc[~P["nota"].isna(), "curso"].unique())
    else:
        cursos_com_nota = set()

    unique_cursos = P["curso"].dropna().unique()
    for c in unique_cursos:
        k = curso_norm(c)
        has = None

        # 1) CSV é sempre prioridade
        if k in PRODUTOS_PROVA_MAP:
            has = bool(PRODUTOS_PROVA_MAP[k])
        else:
            # 2) fallback: heurística pelas notas do próprio progresso.xlsx
            if c in cursos_com_nota:
                has = True
            else:
                has = False

        CURSO_HAS_PROVA_USE[k] = has


def curso_tem_prova(nome_curso: str) -> bool:
    """Consulta se o curso tem prova:
       1) Usa sempre o CSV (PRODUTOS_PROVA_MAP) quando existir
       2) Senão, usa heurística de notas (CURSO_HAS_PROVA_USE)
    """
    if not nome_curso:
        return False
    k = curso_norm(nome_curso)

    if k in PRODUTOS_PROVA_MAP:
        return bool(PRODUTOS_PROVA_MAP[k])

    if k in CURSO_HAS_PROVA_USE:
        return bool(CURSO_HAS_PROVA_USE[k])

    return False


def is_curso_sem_prova(nome_curso: str) -> bool:
    return not curso_tem_prova(nome_curso)


# cursos com prova efetivos (apenas cursos presentes em P)
cursos_com_prova = {
    c for c in (P["curso"].dropna().unique() if "curso" in P.columns else [])
    if curso_tem_prova(c)
}


# ================== 5) Normalizar trilhas e curso → trilha ==================
if len(P) and "trilha" in P.columns:
    P["trilha"] = P["trilha"].apply(
        lambda x: norm_text(x) if pd.notna(x) and str(x).strip() else "(sem trilha)"
    )

if len(T) and "trilha" in T.columns:
    T["trilha"] = T["trilha"].apply(
        lambda x: norm_text(x) if pd.notna(x) and str(x).strip() else "(sem trilha)"
    )
elif len(T) and "trilha_origem" in T.columns:
    T["trilha_origem"] = T["trilha_origem"].apply(
        lambda x: norm_text(x) if pd.notna(x) and str(x).strip() else "(sem trilha)"
    )

m1 = P[["curso","trilha"]].dropna().drop_duplicates() if "trilha" in P else pd.DataFrame(columns=["curso","trilha"])
m2 = pd.DataFrame(columns=["curso","trilha"])
if len(T):
    m2 = T[T["curso"]!=""][["curso"]].drop_duplicates()
    if len(m2):
        m2["trilha"] = m2["curso"].map(pd.Series(P.set_index("curso")["trilha"]).to_dict())

curso_to_trilha = (
    pd.concat([m1, m2], ignore_index=True)
      .drop_duplicates("curso")
      .set_index("curso")["trilha"]
      .to_dict()
)

if len(P):
    P["trilha"] = P["curso"].map(lambda c: curso_to_trilha.get(c, "(sem trilha)"))

if len(T):
    mapped = T["curso"].apply(lambda c: curso_to_trilha.get(c, None) if c!="" else None)
    T["trilha"] = np.where(
        T["curso"]!="",
        mapped.fillna("(sem trilha)"),
        T["trilha_origem"].fillna("(sem trilha)")
    )


# ================== 6) Meses e snapshots ==================
meses = sorted(set(P["ym"].dropna().unique()).union(set(T["ym"].dropna().unique())))
if not meses:
    raise SystemExit("Sem meses válidos.")

P_sorted = P.copy()
P_sorted["_dt"] = pd.to_datetime(P_sorted["atualizado_em"], errors="coerce")

# snapshot global (último registro de cada aluno/curso)
snapshot_global = (
    P_sorted.sort_values(["aluno","curso","_dt"])
            .groupby(["aluno","curso"], as_index=False).tail(1)
)
snapshot_global["trilha"] = snapshot_global["curso"].map(lambda c: curso_to_trilha.get(c, "(sem trilha)"))

# último progresso/nota DENTRO do mês (==m) por aluno/curso
last_in_month = {}
for m in meses:
    dfm = P_sorted[P_sorted["ym"] == m].sort_values(["aluno","curso","_dt"])
    if len(dfm):
        last = dfm.groupby(["aluno","curso"], as_index=False).tail(1).copy()
        last.loc[:, "trilha"] = last["curso"].map(lambda c: curso_to_trilha.get(c, "(sem trilha)"))
        last_in_month[m] = last[["aluno","curso","trilha","progresso","nota"]].copy()
    else:
        last_in_month[m] = pd.DataFrame(columns=["aluno","curso","trilha","progresso","nota"])

# snapshot cumulativo ≤m para cada mês
last_by_month_snapshot = {}
for m in meses:
    dfm = P_sorted[(P_sorted["ym"].isna()) | (P_sorted["ym"] <= m)].sort_values(["aluno","curso","_dt"])
    last = dfm.groupby(["aluno","curso"], as_index=False).tail(1).copy()
    last.loc[:, "trilha"] = last["curso"].map(lambda c: curso_to_trilha.get(c, "(sem trilha)"))
    last_by_month_snapshot[m] = last[["aluno","curso","trilha","progresso","nota"]].copy()


# ================== 7) Séries por trilha ==================
trilhas = sorted(
    set((P["trilha"] if "trilha" in P else pd.Series(dtype=str)).dropna().map(norm_text).unique())
    .union(
    set((T["trilha"] if "trilha" in T else pd.Series(dtype=str)).dropna().map(norm_text).unique()))
) or ["(sem trilha)"]

series_trilhas = {}
for tr in trilhas:
    progMed, hadUpdate, pontos_m, concl, ativos_m = [], [], [], [], []
    nota_med_tr, taxa_com_prova_tr, taxa_aprov_tr = [], [], []

    for m in meses:
        # snapshot ≤m para progresso/nota
        snap = last_by_month_snapshot[m]
        snap_tr = snap[snap["trilha"] == tr]

        # progresso médio por curso dentro da trilha
        if len(snap_tr):
            by_course = snap_tr.groupby("curso", as_index=False)["progresso"].mean()
            val = float(by_course["progresso"].mean())
        else:
            val = 0.0
        progMed.append(round(val, 2))

        # houve atualização de progresso no mês?
        lm = last_in_month[m]
        lm_tr = lm[lm["trilha"] == tr]
        had = bool(len(lm_tr))
        hadUpdate.append(had)

        # pontos / conclusões no mês
        tm = T[(T["ym"] == m) & (T["trilha"] == tr)]
        pontos_m.append(int(np.rint(tm["pontos"].sum())) if len(tm) else 0)
        concl.append(int(tm["tipo"].str.contains("curso_concluido", case=False, na=False).sum()) if len(tm) else 0)

        # alunos ativos no mês (progresso no mês OU pontos no mês naquela trilha)
        alunos_prog = set(lm_tr["aluno"].unique())
        alunos_pts  = set(tm["aluno"].unique())
        ativos_m.append(len(alunos_prog.union(alunos_pts)))

        # NOTA por trilha/mês (somente cursos COM prova)
        if len(snap_tr):
            snap_nota_tr = snap_tr[
                snap_tr["nota"].notna() &
                snap_tr["curso"].isin(cursos_com_prova)
            ]
            notas_vals = snap_nota_tr["nota"].dropna().tolist()
            total_matriculas_tr = len(snap_tr)
            if notas_vals:
                media = float(np.mean(notas_vals))
                qtd_provas = len(notas_vals)
                aprovados = sum(v >= NOTA_APROVACAO for v in notas_vals)
                taxa_aprov = aprovados / qtd_provas
                taxa_cp = qtd_provas / max(total_matriculas_tr, 1)
            else:
                media = 0.0
                taxa_aprov = 0.0
                taxa_cp = 0.0
        else:
            media = 0.0
            taxa_aprov = 0.0
            taxa_cp = 0.0

        nota_med_tr.append(round(media, 2))
        taxa_com_prova_tr.append(round(taxa_cp * 100.0, 1))
        taxa_aprov_tr.append(round(taxa_aprov * 100.0, 1))

    # delta mês a mês de progresso
    progDelta = []
    for i, _ in enumerate(meses):
        if i == 0:
            progDelta.append(0.0)
            continue
        delta = round(progMed[i] - progMed[i-1], 2)
        progDelta.append(delta if hadUpdate[i] else 0.0)

    series_trilhas[tr] = {
        "meses": meses,
        "progMed": progMed,
        "progDelta": progDelta,
        "progHadUpdate": hadUpdate,
        "pontos": pontos_m,
        "conclusoes": concl,
        "ativos": ativos_m,
        "notaMedia": nota_med_tr,
        "taxaComProva": taxa_com_prova_tr,
        "taxaAprovacao": taxa_aprov_tr,
    }


# ================== 8) Top cursos — visão por trilha ==================
lib_trilha = P.groupby(["trilha","curso"], as_index=False)["aluno"].nunique()\
              .rename(columns={"aluno":"alunos_liberados"})

prog_trilha = snapshot_global.groupby(["trilha","curso"], as_index=False)\
                             .agg(progresso_medio=("progresso","mean"))
prog_trilha["progresso_medio"] = prog_trilha["progresso_medio"].round(2)

if "nota" in snapshot_global.columns:
    g_nt = snapshot_global.groupby(["trilha","curso"], as_index=False)
    nota_trilha = g_nt.agg(
        nota_media=("nota", lambda x: float(np.nanmean(x)) if x.notna().any() else np.nan),
        provas_realizadas=("nota", lambda x: int(x.notna().sum())),
        aprovados=("nota", lambda x: int((x >= NOTA_APROVACAO).sum()))
    )
    nota_trilha["taxa_aprovacao"] = nota_trilha.apply(
        lambda r: (r["aprovados"]/r["provas_realizadas"]*100.0) if r["provas_realizadas"]>0 else 0.0,
        axis=1
    )
    nota_trilha["nota_media"] = nota_trilha["nota_media"].round(2)
else:
    nota_trilha = pd.DataFrame(columns=["trilha","curso","nota_media","provas_realizadas","aprovados","taxa_aprovacao"])

# pontos não manuais por trilha/curso
pts_trilha  = (T[(T["is_manual"]==False) & (T["curso"]!="")]
               .groupby(["trilha","curso"], as_index=False)["pontos"].sum()
               .rename(columns={"pontos":"pontos_total"}))

top_cursos_global = {}
for tr in trilhas:
    base = (lib_trilha[lib_trilha["trilha"]==tr]
              .merge(pts_trilha[pts_trilha["trilha"]==tr], on=["trilha","curso"], how="left")
              .merge(prog_trilha[prog_trilha["trilha"]==tr], on=["trilha","curso"], how="left")
              .merge(nota_trilha[nota_trilha["trilha"]==tr], on=["trilha","curso"], how="left")
              .fillna({
                  "pontos_total":0,
                  "progresso_medio":0,
                  "nota_media":0,
                  "provas_realizadas":0,
                  "aprovados":0,
                  "taxa_aprovacao":0
              }))
    base["pontos_total"] = np.rint(base["pontos_total"]).astype(int)

    # flag de prova por curso (para o JSON)
    base["tem_prova"] = base["curso"].apply(curso_tem_prova)

    base = base.sort_values(
        ["progresso_medio","alunos_liberados","pontos_total"],
        ascending=[False, False, False]
    )

    rows = base[[
        "curso",
        "alunos_liberados",
        "progresso_medio",
        "pontos_total",
        "nota_media",
        "provas_realizadas",
        "taxa_aprovacao",
        "tem_prova"
    ]].to_dict(orient="records")
    top_cursos_global[tr] = rows


# ================== 9) Top cursos — visão geral (overall) ==================
lib_all = P.groupby(["curso"], as_index=False)["aluno"].nunique().rename(columns={"aluno":"alunos_liberados"})
prog_all = snapshot_global.groupby(["curso"], as_index=False).agg(progresso_medio=("progresso","mean"))
prog_all["progresso_medio"] = prog_all["progresso_medio"].round(2)

if "nota" in snapshot_global.columns:
    g_all = snapshot_global.groupby(["curso"], as_index=False)
    nota_all = g_all.agg(
        nota_media=("nota", lambda x: float(np.nanmean(x)) if x.notna().any() else np.nan),
        provas_realizadas=("nota", lambda x: int(x.notna().sum())),
        aprovados=("nota", lambda x: int((x >= NOTA_APROVACAO).sum()))
    )
    nota_all["taxa_aprovacao"] = nota_all.apply(
        lambda r: (r["aprovados"]/r["provas_realizadas"]*100.0) if r["provas_realizadas"]>0 else 0.0,
        axis=1
    )
    nota_all["nota_media"] = nota_all["nota_media"].round(2)
else:
    nota_all = pd.DataFrame(columns=["curso","nota_media","provas_realizadas","aprovados","taxa_aprovacao"])

pts_all  = (T[(T["is_manual"]==False) & (T["curso"]!="")]
            .groupby(["curso"], as_index=False)["pontos"].sum()
            .rename(columns={"pontos":"pontos_total"}))

overall_base = (lib_all
    .merge(pts_all, on="curso", how="left")
    .merge(prog_all, on="curso", how="left")
    .merge(nota_all, on="curso", how="left")
    .fillna({
        "pontos_total":0,
        "progresso_medio":0,
        "nota_media":0,
        "provas_realizadas":0,
        "aprovados":0,
        "taxa_aprovacao":0
    }))
overall_base["pontos_total"] = np.rint(overall_base["pontos_total"]).astype(int)

# flag de prova por curso também na visão geral
overall_base["tem_prova"] = overall_base["curso"].apply(curso_tem_prova)

overall_base = overall_base.sort_values(
    ["progresso_medio","alunos_liberados","pontos_total"],
    ascending=[False,False,False]
)

manual_pts_total = int(np.rint(T[T["is_manual"] == True]["pontos"].sum())) if len(T) else 0
manual_users = int(T[T["is_manual"] == True]["aluno"].nunique()) if len(T) else 0
overall_top_cursos = overall_base[[
    "curso",
    "alunos_liberados",
    "progresso_medio",
    "pontos_total",
    "nota_media",
    "provas_realizadas",
    "taxa_aprovacao",
    "tem_prova"
]].to_dict(orient="records")

if manual_pts_total > 0 or manual_users > 0:
    overall_top_cursos = ([{
        "curso": "Pontos Manuais",
        "alunos_liberados": manual_users,
        "progresso_medio": "Não se aplica",
        "pontos_total": manual_pts_total,
        "nota_media": "Não se aplica",
        "provas_realizadas": "Não se aplica",
        "taxa_aprovacao": "Não se aplica",
        "tem_prova": False
    }] + overall_top_cursos)


# ================== 10) Séries gerais (overall) ==================
prog_o, pontos_o, concl_o, ativos_o = [], [], [], []
nota_media_o, taxa_com_prova_o, taxa_aprov_o = [], [], []

for m in meses:
    # progresso / nota (snapshot ≤m)
    snap = last_by_month_snapshot[m]
    vals = snap["progresso"].dropna().tolist()
    prog_o.append(round(float(np.mean(vals)) if vals else 0.0, 2))

    # pontos no mês (inclui manuais)
    pontos_o.append(int(np.rint(T[T["ym"]==m]["pontos"].sum())))

    # conclusões no mês
    concl_o.append(int(T[(T["ym"]==m) & (T["tipo"].str.contains("curso_concluido", case=False, na=False))].shape[0]))

    # alunos ativos no mês (progresso OU pontos em qualquer trilha)
    lm = last_in_month[m]
    prog_alunos = set(lm["aluno"].unique())
    pts_m = T[T["ym"]==m]
    pts_alunos = set(pts_m["aluno"].unique())
    ativos_o.append(len(prog_alunos.union(pts_alunos)))

    # nota geral no mês (somente cursos COM prova)
    total_matriculas = len(snap)
    snap_nota = snap[
        snap["nota"].notna() &
        snap["curso"].isin(cursos_com_prova)
    ]
    notas_vals = snap_nota["nota"].dropna().tolist()
    if notas_vals:
        media = float(np.mean(notas_vals))
        qtd_provas = len(notas_vals)
        aprovados = sum(v >= NOTA_APROVACAO for v in notas_vals)
        taxa_aprov = aprovados / qtd_provas
        taxa_cp = qtd_provas / max(total_matriculas, 1)
    else:
        media = 0.0
        taxa_aprov = 0.0
        taxa_cp = 0.0

    nota_media_o.append(round(media, 2))
    taxa_com_prova_o.append(round(taxa_cp * 100.0, 1))
    taxa_aprov_o.append(round(taxa_aprov * 100.0, 1))


# ================== 11) Tabelas de alunos por mês (students + manuals) ==================
# Pontos não manuais por aluno/curso/trilha (base mensal)
pts_a_c_t_mes = (
    T[(T["is_manual"] == False) & (T["curso"] != "")]
      .groupby(["ym", "aluno", "curso", "trilha"], as_index=False)["pontos"]
      .sum()
      .rename(columns={"pontos": "pontos_mes"})
)

if len(pts_a_c_t_mes):
    pts_a_c_t_mes["ym"] = pd.Categorical(pts_a_c_t_mes["ym"], categories=meses, ordered=True)
    pts_a_c_t_mes = pts_a_c_t_mes.sort_values(["aluno","curso","trilha","ym"])
    pts_a_c_t_mes["pontos_cum"] = pts_a_c_t_mes.groupby(["aluno","curso","trilha"])["pontos_mes"].cumsum()
else:
    pts_a_c_t_mes["pontos_cum"] = pd.Series(dtype=float)

manuals_a_mes = (
    T[T["is_manual"] == True]
      .groupby(["ym", "aluno"], as_index=False)["pontos"]
      .sum()
      .rename(columns={"pontos": "pontos_manua_mes"})
)

if len(manuals_a_mes):
    manuals_a_mes["ym"] = pd.Categorical(manuals_a_mes["ym"], categories=meses, ordered=True)
    manuals_a_mes = manuals_a_mes.sort_values(["aluno","ym"])
    manuals_a_mes["pontos_manua_cum"] = manuals_a_mes.groupby(["aluno"])["pontos_manua_mes"].cumsum()
else:
    manuals_a_mes["pontos_manua_cum"] = pd.Series(dtype=float)

students = {}
manuals  = {}

for m in meses:
    snap = last_by_month_snapshot[m].copy()  # aluno,curso,trilha,progresso,nota

    if len(pts_a_c_t_mes):
        pts_cum_le_m = pts_a_c_t_mes[pts_a_c_t_mes["ym"] <= m]
        if len(pts_cum_le_m):
            pts_cum_last = (
                pts_cum_le_m
                .sort_values(["aluno","curso","trilha","ym"])
                .groupby(["aluno","curso","trilha"], as_index=False)
                .tail(1)[["aluno","curso","trilha","pontos_cum"]]
            )
        else:
            pts_cum_last = pd.DataFrame(columns=["aluno","curso","trilha","pontos_cum"])
    else:
        pts_cum_last = pd.DataFrame(columns=["aluno","curso","trilha","pontos_cum"])

    base = snap.merge(pts_cum_last, on=["aluno","curso","trilha"], how="outer")

    if "progresso" not in base.columns:
        base["progresso"] = 0.0
    base["progresso"] = base["progresso"].fillna(0.0)

    if "pontos_cum" not in base.columns:
        base["pontos_cum"] = 0
    base["pontos_cum"] = base["pontos_cum"].fillna(0)

    if "nota" not in base.columns:
        base["nota"] = np.nan

    def resolve_nota(row):
        curso = row.get("curso") or ""
        nota  = row.get("nota")

        # 1) cursos SEM prova (definidos via CSV + fallback de notas)
        if is_curso_sem_prova(curso):
            return "Sem prova"

        # 2) cursos COM prova, mas esse aluno ainda não fez
        if nota is None or (isinstance(nota, float) and math.isnan(nota)):
            return "Não realizado"

        # 3) nota numérica
        try:
            v = fnum(nota)
            if math.isnan(v):
                return "Não realizado"
            return str(int(round(v)))
        except Exception:
            return "Não realizado"

    base["nota_prova"] = base.apply(resolve_nota, axis=1)
    base["tem_prova"] = base["curso"].apply(curso_tem_prova)

    base = base.rename(columns={"pontos_cum": "pontos_mes"})
    out_cols = ["aluno","curso","trilha","progresso","nota_prova","pontos_mes","tem_prova"]
    base = base[out_cols].sort_values(["trilha","aluno","curso"], kind="stable")
    students[m] = base.to_dict(orient="records")

    if len(manuals_a_mes):
        man_le_m = manuals_a_mes[manuals_a_mes["ym"] <= m]
        if len(man_le_m):
            man_last = (
                man_le_m.sort_values(["aluno","ym"])
                        .groupby(["aluno"], as_index=False)
                        .tail(1)[["aluno","pontos_manua_cum"]]
            )
            man_last["pontos_manua_cum"] = np.rint(man_last["pontos_manua_cum"]).astype(int)
        else:
            man_last = pd.DataFrame(columns=["aluno","pontos_manua_cum"])
    else:
        man_last = pd.DataFrame(columns=["aluno","pontos_manua_cum"])

    if len(man_last):
        man_last = man_last.rename(columns={"pontos_manua_cum": "pontos_manua_mes"})
    manuals[m] = man_last.sort_values(["aluno"], kind="stable").to_dict(orient="records")


# ================== 12) Montar saída final ==================
out = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "meses": meses,
    "overall": {
        "meses": meses,
        "progMed": prog_o,
        "pontos": pontos_o,
        "conclusoes": concl_o,
        "ativos": ativos_o,
        "notaMedia": nota_media_o,
        "taxaComProva": taxa_com_prova_o,
        "taxaAprovacao": taxa_aprov_o,
    },
    "overall_top_cursos": overall_top_cursos,
    "trilhas": {
        tr: {
            "series": series_trilhas[tr],
            "top_cursos_global": top_cursos_global.get(tr, [])
        } for tr in trilhas
    },
    "students": students,
    "manuals": manuals
}

with open("data_trilhas.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

print("✅ data_trilhas.json gerado com sucesso — usando *sempre* produtos_provas.csv como fonte principal para cursos com/sem prova.")
