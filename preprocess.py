# preprocess.py — gera data_trilhas.json a partir de progresso.xlsx e pontos.xlsx
# Regras aplicadas:
# - "manual" identificado exclusivamente pela coluna TIPO (case-insensitive)
# - Progresso da trilha no mês = média das médias de progresso por curso (no mês)
# - Sem rateio de pontos entre cursos
# - Linha "Pontos Manuais" apenas no Top cursos — visão geral (overall)
# - Valores de pontos sempre inteiros (arredondados)

import pandas as pd
import numpy as np
import json, re
from datetime import datetime, timezone

# ----------------- helpers -----------------
def norm_text(s):
    s = "" if s is None else str(s)
    s = re.sub(r"\s+", " ", s.strip())
    return s

def norm_key(s): 
    return norm_text(s).lower()

def find_col(df, candidates):
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

# ----------------- 1) ler planilhas -----------------
progresso = pd.read_excel("progresso.xlsx")
pontos    = pd.read_excel("pontos.xlsx")
progresso.columns = [norm_text(c) for c in progresso.columns]
pontos.columns    = [norm_text(c) for c in pontos.columns]

# variações de colunas
P_ALUNO = find_col(progresso, ["aluno","nome do aluno","usuário","usuario","nome"])
P_CURSO = find_col(progresso, ["curso","produto","formação","formacao"])
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

# ----------------- 2) canonicalizar linhas -----------------
def map_progresso(df):
    rows = []
    for _, r in df.iterrows():
        aluno = r.get(P_ALUNO) if P_ALUNO else None
        curso = r.get(P_CURSO) if P_CURSO else None
        trilha = r.get(P_TRILHA) if P_TRILHA else None
        prog = r.get(P_PROG) if P_PROG else None
        dt   = r.get(P_DT)   if P_DT   else None
        nota = r.get(P_NOTA) if P_NOTA else None

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

def map_pontos(df):
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

# Cursos que efetivamente têm prova (alguma linha com nota numérica)
if "nota" in P.columns:
    cursos_com_prova = set(P.loc[~P["nota"].isna(), "curso"].unique())
else:
    cursos_com_prova = set()


# GARANTE trilhas normalizadas (sem espaços extras, mesmo case)
if len(P) and "trilha" in P.columns:
    P["trilha"] = P["trilha"].apply(
        lambda x: norm_text(x) if pd.notna(x) and str(x).strip() else "(sem trilha)"
    )

# Em T, nesta altura AINDA não existe T["trilha"]; normalize a coluna de origem
if len(T) and "trilha" in T.columns:
    T["trilha"] = T["trilha"].apply(
        lambda x: norm_text(x) if pd.notna(x) and str(x).strip() else "(sem trilha)"
    )
elif len(T) and "trilha_origem" in T.columns:
    T["trilha_origem"] = T["trilha_origem"].apply(
        lambda x: norm_text(x) if pd.notna(x) and str(x).strip() else "(sem trilha)"
    )

# ----------------- 3) curso → trilha -----------------
m1 = P[["curso","trilha"]].dropna().drop_duplicates() if "trilha" in P else pd.DataFrame(columns=["curso","trilha"])
m2 = pd.DataFrame(columns=["curso","trilha"])
if len(T):
    m2 = T[T["curso"]!=""][["curso"]].drop_duplicates()
    if len(m2):
        m2["trilha"] = m2["curso"].map(pd.Series(P.set_index("curso")["trilha"]).to_dict())

curso_to_trilha = pd.concat([m1, m2], ignore_index=True)\
                    .drop_duplicates("curso")\
                    .set_index("curso")["trilha"].to_dict()

if len(P):
    P["trilha"] = P["curso"].map(lambda c: curso_to_trilha.get(c, "(sem trilha)"))

if len(T):
    mapped = T["curso"].apply(lambda c: curso_to_trilha.get(c, None) if c!="" else None)
    T["trilha"] = np.where(
        T["curso"]!="",
        mapped.fillna("(sem trilha)"),
        T["trilha_origem"].fillna("(sem trilha)")
    )

# ----------------- 4) meses e snapshots -----------------
meses = sorted(set(P["ym"].dropna().unique()).union(set(T["ym"].dropna().unique())))
if not meses:
    raise SystemExit("Sem meses válidos.")

P_sorted = P.copy()
P_sorted["_dt"] = pd.to_datetime(P_sorted["atualizado_em"], errors="coerce")

# snapshot cumulativo (≤ mês) — útil para KPIs, mas NÃO será usado no gráfico
snapshot_global = (
    P_sorted.sort_values(["aluno","curso","_dt"])
            .groupby(["aluno","curso"], as_index=False).tail(1)
)
snapshot_global["trilha"] = snapshot_global["curso"].map(lambda c: curso_to_trilha.get(c, "(sem trilha)"))

# 4a) Último progresso no MÊS (== mês) para cada (aluno, curso)
last_in_month = {}
for m in meses:
    dfm = P_sorted[P_sorted["ym"] == m].sort_values(["aluno","curso","_dt"])
    if len(dfm):
        last = dfm.groupby(["aluno","curso"], as_index=False).tail(1).copy()
        last.loc[:, "trilha"] = last["curso"].map(lambda c: curso_to_trilha.get(c, "(sem trilha)"))
        last_in_month[m] = last[["aluno","curso","trilha","progresso","nota"]].copy()
    else:
        last_in_month[m] = pd.DataFrame(columns=["aluno","curso","trilha","progresso","nota"])

# 4b) Snapshot cumulativo por mês (≤ mês) — mantém para KPIs
last_by_month_snapshot = {}
for m in meses:
    dfm = P_sorted[(P_sorted["ym"].isna()) | (P_sorted["ym"] <= m)].sort_values(["aluno","curso","_dt"])
    
    last = dfm.groupby(["aluno","curso"], as_index=False).tail(1).copy()
    last.loc[:, "trilha"] = last["curso"].map(lambda c: curso_to_trilha.get(c, "(sem trilha)"))
    last_by_month_snapshot[m] = last[["aluno","curso","trilha","progresso","nota"]].copy()


# ----------------- 5) séries por trilha -----------------
# Estratégia:
# - Progresso da trilha no mês = média das médias de progresso por CURSO
#   usando o snapshot cumulativo (≤ mês) -> sempre há um valor.
# - progHadUpdate[m] = True se EXISTE atualização no mês (== m) para a trilha.

trilhas = sorted(
    set((P["trilha"] if "trilha" in P else pd.Series(dtype=str)).dropna().map(norm_text).unique())
    .union(
    set((T["trilha"] if "trilha" in T else pd.Series(dtype=str)).dropna().map(norm_text).unique()))
) or ["(sem trilha)"]

series_trilhas = {}
for tr in trilhas:
    progMed, hadUpdate, pontos_m, concl, ativos_m = [], [], [], [], []
    for m in meses:
        # Snapshot ≤ m
        snap = last_by_month_snapshot[m]
        snap_tr = snap[snap["trilha"] == tr]
        if len(snap_tr):
            by_course = snap_tr.groupby("curso", as_index=False)["progresso"].mean()
            val = float(by_course["progresso"].mean())
        else:
            val = 0.0
        progMed.append(round(val, 2))

        # Houve atualização no mês?
        lm = last_in_month[m]
        had = bool(len(lm[lm["trilha"] == tr]))
        hadUpdate.append(had)

        # Pontos / conclusões (== m)
        tm = T[(T["ym"] == m) & (T["trilha"] == tr)]
        pontos_m.append(int(np.rint(tm["pontos"].sum())) if len(tm) else 0)
        concl.append(int(tm["tipo"].str.contains("curso_concluido", case=False, na=False).sum()) if len(tm) else 0)

        # Ativos (snapshot ≤ m)
        ativos_m.append(int(snap_tr["aluno"].nunique()))

    # Δ mês a mês (pontos percentuais) – usa diferenças do snapshot e zera quando não houve update
    progDelta = []
    for i, _ in enumerate(meses):
        if i == 0:
            progDelta.append(0.0)
            continue
        delta = round(progMed[i] - progMed[i-1], 2)
        progDelta.append(delta if hadUpdate[i] else 0.0)

    series_trilhas[tr] = {
        "meses": meses,
        "progMed": progMed,             # snapshot
        "progDelta": progDelta,         # variação mês a mês (p.p.)
        "progHadUpdate": hadUpdate,     # flag p/ styling
        "pontos": pontos_m,
        "conclusoes": concl,
        "ativos": ativos_m
    }



# ----------------- 6) Top cursos — visão geral (somente pontos NÃO manuais) -----------------
# alunos com curso liberado por trilha/curso
lib_trilha = P.groupby(["trilha","curso"], as_index=False)["aluno"].nunique()\
              .rename(columns={"aluno":"alunos_liberados"})

# progresso médio (snapshot global por trilha/curso)
prog_trilha = snapshot_global.groupby(["trilha","curso"], as_index=False)\
                             .agg(progresso_medio=("progresso","mean"))
prog_trilha["progresso_medio"] = prog_trilha["progresso_medio"].round(2)

# pontos por trilha/curso EXCLUINDO manuais
pts_trilha  = (T[(T["is_manual"]==False) & (T["curso"]!="")]
               .groupby(["trilha","curso"], as_index=False)["pontos"].sum()
               .rename(columns={"pontos":"pontos_total"}))

top_cursos_global = {}
for tr in trilhas:
    base = (lib_trilha[lib_trilha["trilha"]==tr]
              .merge(pts_trilha[pts_trilha["trilha"]==tr], on=["trilha","curso"], how="left")
              .merge(prog_trilha[prog_trilha["trilha"]==tr], on=["trilha","curso"], how="left")
              .fillna({"pontos_total":0,"progresso_medio":0}))
    base["pontos_total"] = np.rint(base["pontos_total"]).astype(int)
    base = base.sort_values(["progresso_medio","alunos_liberados","pontos_total"], ascending=[False, False, False])

    rows = base[["curso","alunos_liberados","progresso_medio","pontos_total"]].to_dict(orient="records")
    top_cursos_global[tr] = rows

# visão geral (sem trilha)
lib_all = P.groupby(["curso"], as_index=False)["aluno"].nunique().rename(columns={"aluno":"alunos_liberados"})
prog_all = snapshot_global.groupby(["curso"], as_index=False).agg(progresso_medio=("progresso","mean"))
prog_all["progresso_medio"] = prog_all["progresso_medio"].round(2)
pts_all  = (T[(T["is_manual"]==False) & (T["curso"]!="")]
            .groupby(["curso"], as_index=False)["pontos"].sum()
            .rename(columns={"pontos":"pontos_total"}))
overall_base = (lib_all.merge(pts_all, on="curso", how="left")
                      .merge(prog_all, on="curso", how="left")
                      .fillna({"pontos_total":0,"progresso_medio":0}))
overall_base["pontos_total"] = np.rint(overall_base["pontos_total"]).astype(int)
overall_base = overall_base.sort_values(["progresso_medio","alunos_liberados","pontos_total"], ascending=[False,False,False])

# Linha "Pontos Manuais" no topo (overall)
manual_pts_total = int(np.rint(T[T["is_manual"] == True]["pontos"].sum())) if len(T) else 0
manual_users = int(T[T["is_manual"] == True]["aluno"].nunique()) if len(T) else 0
overall_top_cursos = overall_base[["curso","alunos_liberados","progresso_medio","pontos_total"]].to_dict(orient="records")
if manual_pts_total > 0 or manual_users > 0:
    overall_top_cursos = ([{
        "curso": "Pontos Manuais",
        "alunos_liberados": manual_users,
        "progresso_medio": "Não se aplica",
        "pontos_total": manual_pts_total
    }] + overall_top_cursos)

# ----------------- 7) séries gerais (overall) -----------------
prog_o, pontos_o, concl_o, ativos_o = [], [], [], []
for m in meses:
    last = last_by_month_snapshot[m]
    vals = last["progresso"].dropna().tolist()
    prog_o.append(round(float(np.mean(vals)) if vals else 0.0, 2))
    pontos_o.append(int(np.rint(T[T["ym"]==m]["pontos"].sum())))  # inclui manuais
    concl_o.append(int(T[(T["ym"]==m) & (T["tipo"].str.contains("curso_concluido", case=False, na=False))].shape[0]))
    ativos_o.append(int(last["aluno"].nunique()))

# ----------------- 8) alunos por mês (AGORA: pontos ACUMULADOS até o mês) -----------------
# 8.1) Pontos NÃO manuais por aluno/curso/trilha (com curso definido) — base mensal
pts_a_c_t_mes = (
    T[(T["is_manual"] == False) & (T["curso"] != "")]
      .groupby(["ym", "aluno", "curso", "trilha"], as_index=False)["pontos"]
      .sum()
      .rename(columns={"pontos": "pontos_mes"})
)

# 8.2) Gerar série ACUMULADA (cumsum) respeitando a ordem de 'meses'
if len(pts_a_c_t_mes):
    pts_a_c_t_mes["ym"] = pd.Categorical(pts_a_c_t_mes["ym"], categories=meses, ordered=True)
    pts_a_c_t_mes = pts_a_c_t_mes.sort_values(["aluno","curso","trilha","ym"])
    pts_a_c_t_mes["pontos_cum"] = pts_a_c_t_mes.groupby(["aluno","curso","trilha"])["pontos_mes"].cumsum()
else:
    pts_a_c_t_mes["pontos_cum"] = pd.Series(dtype=float)

# 8.3) PONTOS MANUAIS — agregados por aluno (sem curso/trilha), base mensal
manuals_a_mes = (
    T[T["is_manual"] == True]
      .groupby(["ym", "aluno"], as_index=False)["pontos"]
      .sum()
      .rename(columns={"pontos": "pontos_manua_mes"})
)

# 8.4) Série ACUMULADA (cumsum) para manuais
if len(manuals_a_mes):
    manuals_a_mes["ym"] = pd.Categorical(manuals_a_mes["ym"], categories=meses, ordered=True)
    manuals_a_mes = manuals_a_mes.sort_values(["aluno","ym"])
    manuals_a_mes["pontos_manua_cum"] = manuals_a_mes.groupby(["aluno"])["pontos_manua_mes"].cumsum()
else:
    manuals_a_mes["pontos_manua_cum"] = pd.Series(dtype=float)

students = {}
manuals   = {}

for m in meses:
    # Snapshot ≤ m (progresso mais recente por aluno-curso-trilha)
    snap = last_by_month_snapshot[m].copy()  # cols: aluno, curso, trilha, progresso

    # 8.5) Pegar, para cada (aluno,curso,trilha), o último registro ACUMULADO cujo ym <= m
    if len(pts_a_c_t_mes):
        pts_cum_le_m = pts_a_c_t_mes[pts_a_c_t_mes["ym"] <= m]
        if len(pts_cum_le_m):
            # pega a última linha por chave (aluno,curso,trilha)
            pts_cum_last = (pts_cum_le_m
                            .sort_values(["aluno","curso","trilha","ym"])
                            .groupby(["aluno","curso","trilha"], as_index=False)
                            .tail(1)[["aluno","curso","trilha","pontos_cum"]])
        else:
            pts_cum_last = pd.DataFrame(columns=["aluno","curso","trilha","pontos_cum"])
    else:
        pts_cum_last = pd.DataFrame(columns=["aluno","curso","trilha","pontos_cum"])

    # 8.6) UNIÃO de bases: snapshot (progresso) ⟷ pontos acumulados
    base = snap.merge(pts_cum_last, on=["aluno","curso","trilha"], how="outer")

    # Defaults
    if "progresso" not in base.columns:
        base["progresso"] = 0.0
    base["progresso"] = base["progresso"].fillna(0.0)

    if "pontos_cum" not in base.columns:
        base["pontos_cum"] = 0
    base["pontos_cum"] = base["pontos_cum"].fillna(0)

    # Inteiros nos pontos
    base["pontos_cum"] = np.rint(base["pontos_cum"]).astype(int)

    # -------- NOVO: transformar nota em texto amigável --------
    import math

    if "nota" not in base.columns:
        base["nota"] = np.nan

    def resolve_nota(row):
        curso = row.get("curso") or ""
        nota  = row.get("nota")

        # curso nunca apareceu com nota → não existe prova pra ele
        if not curso or curso not in cursos_com_prova:
            return "Sem prova"

        # curso tem prova, mas esse aluno não tem nota registrada
        if nota is None or (isinstance(nota, float) and math.isnan(nota)):
            return "Não realizado"

        # nota numérica
        try:
            v = fnum(nota)
            return str(int(round(v)))
        except Exception:
            return "Não realizado"

    base["nota_prova"] = base.apply(resolve_nota, axis=1)

    # Seleção/ordenação e saída
    base = base.rename(columns={"pontos_cum": "pontos_mes"})
    out_cols = ["aluno", "curso", "trilha", "progresso", "nota_prova", "pontos_mes"]
    base = base[out_cols].sort_values(["trilha", "aluno", "curso"], kind="stable")
    students[m] = base.to_dict(orient="records")

    # 8.7) Manuais acumulados por aluno (independente de trilha) — linha sintética na tabela de alunos
    if len(manuals_a_mes):
        man_le_m = manuals_a_mes[manuals_a_mes["ym"] <= m]
        if len(man_le_m):
            man_last = (man_le_m.sort_values(["aluno","ym"])
                                  .groupby(["aluno"], as_index=False)
                                  .tail(1)[["aluno","pontos_manua_cum"]])
            man_last["pontos_manua_cum"] = np.rint(man_last["pontos_manua_cum"]).astype(int)
        else:
            man_last = pd.DataFrame(columns=["aluno","pontos_manua_cum"])
    else:
        man_last = pd.DataFrame(columns=["aluno","pontos_manua_cum"])

    # Guardamos com a mesma estrutura anterior (chave 'pontos_manua_mes' no JSON),
    # mas agora contendo o ACUMULADO até o mês.
    if len(man_last):
        man_last = man_last.rename(columns={"pontos_manua_cum": "pontos_manua_mes"})
    manuals[m] = man_last.sort_values(["aluno"], kind="stable").to_dict(orient="records")


# ================== 9) montar saída final ==================
out = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "meses": meses,
    "overall": {
        "meses": meses,
        "progMed": prog_o,
        "pontos": pontos_o,
        "conclusoes": concl_o,
        "ativos": ativos_o
    },
    "overall_top_cursos": overall_top_cursos,
    "trilhas": {
        tr: {
            "series": series_trilhas[tr],
            "top_cursos_global": top_cursos_global.get(tr, [])
        } for tr in trilhas
    },
    "students": students,   # pontuação não-manual por aluno/curso/trilha
    "manuals": manuals      # pontuação manual agregada por aluno (sem curso)
}

# grava o arquivo JSON final
with open("data_trilhas.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

print("✅ data_trilhas.json gerado com sucesso — sem rateio, manual por 'tipo', e progresso de trilha = média das médias por curso.")
