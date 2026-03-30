import json
import re
import unicodedata
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd


NOTA_APROVACAO = 70.0
CARGO_SUGESTAO_MIN_SCORE = 60.0
COMPLETION_PROGRESS = 99.5
TRAINING_EVENT_TYPES = {
    "aula_concluida",
    "curso_iniciado",
    "curso_progresso_50",
    "curso_progresso_75",
    "curso_progresso_90",
    "curso_concluido",
    "prova_aprovada",
}
MILESTONE_EVENT_VALUE = {
    "aula_concluida": 1,
    "curso_iniciado": 1,
    "curso_progresso_50": 50,
    "curso_progresso_75": 75,
    "curso_progresso_90": 90,
    "curso_concluido": 100,
    "prova_aprovada": 100,
}


def norm_text(value):
    text = "" if value is None else str(value)
    return re.sub(r"\s+", " ", text.strip())


def norm_key(value):
    return norm_text(value).lower()


def norm_match(value):
    text = norm_text(value)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def unique_preserve(values):
    seen = set()
    output = []
    for item in values:
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def pick_display_label(values):
    ranked = {}
    first_position = {}
    for index, item in enumerate(values):
        label = norm_text(item)
        if not label:
            continue
        ranked[label] = ranked.get(label, 0) + 1
        first_position.setdefault(label, index)

    if not ranked:
        return ""

    return sorted(
        ranked,
        key=lambda label: (-ranked[label], first_position[label], len(label), label.lower()),
    )[0]


def similarity(a, b):
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def find_exactish_match(value_norm, candidates_norm):
    if not value_norm:
        return None, None
    for candidate in candidates_norm:
        if value_norm == candidate:
            return candidate, "exact"
    for candidate in candidates_norm:
        if len(candidate) >= 8 and (candidate in value_norm or value_norm in candidate):
            return candidate, "contains"
    best = None
    best_ratio = 0.0
    for candidate in candidates_norm:
        ratio = similarity(value_norm, candidate)
        if ratio > best_ratio:
            best_ratio = ratio
            best = candidate
    if best and best_ratio >= 0.88:
        return best, "fuzzy"
    return None, None


def fnum(value, default=0.0):
    try:
        if isinstance(value, str):
            value = value.replace("%", "").replace(",", ".")
        result = float(value)
        return default if np.isnan(result) else result
    except Exception:
        return default


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def sanitize_progress(value):
    return round(clamp(fnum(value, 0.0), 0.0, 100.0), 2)


def is_completed_progress(value):
    return fnum(value, 0.0) >= COMPLETION_PROGRESS


def parse_datetime(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return pd.NaT
    return pd.to_datetime(value, dayfirst=True, errors="coerce")


def ym_from_datetime(value):
    dt = parse_datetime(value)
    if pd.isna(dt):
        return None
    return f"{dt.year:04d}-{dt.month:02d}"


def find_col(df, candidates):
    cols = {norm_key(col): col for col in df.columns}
    for candidate in candidates:
        found = cols.get(norm_key(candidate))
        if found:
            return found
    return None


def parse_flag(value):
    text = norm_match(value)
    return text in {"sim", "s", "yes", "y", "1", "true", "verdadeiro"}


def parse_grade_value(value):
    number = fnum(value, default=np.nan)
    return number if not np.isnan(number) else np.nan


def parse_duration_seconds(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0
    if isinstance(value, pd.Timedelta):
        return int(round(value.total_seconds()))
    if hasattr(value, "hour") and hasattr(value, "minute") and hasattr(value, "second"):
        return int(value.hour) * 3600 + int(value.minute) * 60 + int(value.second)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value < 2:
            return int(round(float(value) * 86400))
        return int(round(float(value)))

    text = norm_text(value)
    if not text:
        return 0
    try:
        return int(round(pd.to_timedelta(text).total_seconds()))
    except Exception:
        pass

    parts = text.split(":")
    if len(parts) == 3 and all(part.isdigit() for part in parts):
        hours, minutes, seconds = [int(part) for part in parts]
        return hours * 3600 + minutes * 60 + seconds
    if len(parts) == 2 and all(part.isdigit() for part in parts):
        minutes, seconds = [int(part) for part in parts]
        return minutes * 60 + seconds
    return 0


def classify_confidence(top_score, second_score, exact_trail_hits):
    if top_score < CARGO_SUGESTAO_MIN_SCORE:
        return "baixa"
    if second_score <= 0:
        return "alta" if top_score >= 120 else "media"
    ratio = top_score / max(second_score, 1.0)
    if ratio < 1.15:
        return "baixa"
    if ratio >= 1.35 or (top_score >= 160 and exact_trail_hits >= 2):
        return "alta"
    if ratio >= 1.2 or (top_score >= 100 and exact_trail_hits >= 1):
        return "media"
    return "baixa"


def summarize_signal_names(entries, signal_type, limit=4):
    values = []
    seen = set()
    for item in sorted(entries, key=lambda row: (-row["weight"], row["label"])):
        if item["type"] != signal_type:
            continue
        label = item["label"]
        if label in seen:
            continue
        seen.add(label)
        values.append(label)
        if len(values) >= limit:
            break
    return values


def prepare_cargo_profiles(raw_catalog):
    profiles = []
    for item in raw_catalog.get("profiles", []):
        cargo = norm_text(item.get("cargo"))
        if not cargo:
            continue
        profile = {
            "cargo": cargo,
            "source": norm_text(item.get("source")) or "catalogo",
            "trail_exact": unique_preserve([norm_text(v) for v in item.get("trail_exact", []) if norm_text(v)]),
            "trail_keywords": unique_preserve([norm_text(v) for v in item.get("trail_keywords", []) if norm_text(v)]),
            "course_exact": unique_preserve([norm_text(v) for v in item.get("course_exact", []) if norm_text(v)]),
            "course_keywords": unique_preserve([norm_text(v) for v in item.get("course_keywords", []) if norm_text(v)]),
        }
        profile["trail_exact_norm"] = [norm_match(v) for v in profile["trail_exact"]]
        profile["trail_keywords_norm"] = [norm_match(v) for v in profile["trail_keywords"]]
        profile["course_exact_norm"] = [norm_match(v) for v in profile["course_exact"]]
        profile["course_keywords_norm"] = [norm_match(v) for v in profile["course_keywords"]]
        profiles.append(profile)
    return profiles


def infer_cargo_profile(trilhas, cursos, cargo_profiles):
    trilhas = unique_preserve([norm_text(v) for v in trilhas if norm_text(v)])
    cursos = unique_preserve([norm_text(v) for v in cursos if norm_text(v)])
    trilhas_pairs = [(v, norm_match(v)) for v in trilhas]
    cursos_pairs = [(v, norm_match(v)) for v in cursos]

    ranked = []
    for profile in cargo_profiles:
        score = 0.0
        signals = []
        exact_trail_hits = 0
        matched_trail_exact = set()
        matched_trail_keywords = set()
        matched_course_exact = set()
        matched_course_keywords = set()

        for original, value_norm in trilhas_pairs:
            candidate, kind = find_exactish_match(value_norm, profile["trail_exact_norm"])
            if candidate and candidate not in matched_trail_exact:
                matched_trail_exact.add(candidate)
                weight = 90.0 if kind == "exact" else 72.0
                score += weight
                exact_trail_hits += 1
                signals.append({"type": "trilha", "label": original, "weight": weight})

            for keyword, keyword_norm in zip(profile["trail_keywords"], profile["trail_keywords_norm"]):
                if not keyword_norm or keyword_norm in matched_trail_keywords:
                    continue
                if keyword_norm in value_norm:
                    matched_trail_keywords.add(keyword_norm)
                    score += 22.0
                    signals.append({"type": "trilha", "label": original, "weight": 22.0})
                    break

        for original, value_norm in cursos_pairs:
            candidate, kind = find_exactish_match(value_norm, profile["course_exact_norm"])
            if candidate and candidate not in matched_course_exact:
                matched_course_exact.add(candidate)
                weight = 16.0 if kind == "exact" else 12.0
                score += weight
                signals.append({"type": "curso", "label": original, "weight": weight})

            for keyword, keyword_norm in zip(profile["course_keywords"], profile["course_keywords_norm"]):
                if not keyword_norm or keyword_norm in matched_course_keywords:
                    continue
                if keyword_norm in value_norm:
                    matched_course_keywords.add(keyword_norm)
                    score += 10.0
                    signals.append({"type": "curso", "label": original, "weight": 10.0})
                    break

        if score <= 0:
            continue

        ranked.append(
            {
                "cargo": profile["cargo"],
                "source": profile["source"],
                "score": round(score, 1),
                "exact_trail_hits": exact_trail_hits,
                "signals": signals,
            }
        )

    ranked.sort(key=lambda row: (row["score"], row["exact_trail_hits"], len(row["signals"])), reverse=True)
    top = ranked[0] if ranked else None
    second = ranked[1] if len(ranked) > 1 else None

    if not top or top["score"] < CARGO_SUGESTAO_MIN_SCORE:
        return {
            "cargo_sugerido": None,
            "confianca": "baixa",
            "score": round(top["score"], 1) if top else 0.0,
            "cargo_alternativo": second["cargo"] if second else (top["cargo"] if top else None),
            "score_alternativo": round(second["score"], 1) if second else None,
            "trilhas_atribuidas": trilhas,
            "trilhas_sinalizadoras": summarize_signal_names(top["signals"], "trilha") if top else [],
            "cursos_sinalizadores": summarize_signal_names(top["signals"], "curso") if top else [],
            "alternativas": [{"cargo": item["cargo"], "score": item["score"]} for item in ranked[:3]],
            "justificativa": "Sugestao automatica indisponivel: trilhas atribuidas ainda sao amplas ou pouco especificas para um cargo.",
        }

    confidence = classify_confidence(top["score"], second["score"] if second else 0.0, top["exact_trail_hits"])
    trail_signals = summarize_signal_names(top["signals"], "trilha")
    course_signals = summarize_signal_names(top["signals"], "curso")
    rationale = []
    if trail_signals:
        rationale.append(f"Trilhas mais aderentes: {', '.join(trail_signals)}.")
    if course_signals:
        rationale.append(f"Cursos sinalizadores: {', '.join(course_signals)}.")
    if second:
        rationale.append(f"Alternativa proxima: {second['cargo']} ({second['score']:.1f} pts).")

    return {
        "cargo_sugerido": top["cargo"],
        "confianca": confidence,
        "score": round(top["score"], 1),
        "cargo_alternativo": second["cargo"] if second else None,
        "score_alternativo": round(second["score"], 1) if second else None,
        "trilhas_atribuidas": trilhas,
        "trilhas_sinalizadoras": trail_signals,
        "cursos_sinalizadores": course_signals,
        "alternativas": [{"cargo": item["cargo"], "score": item["score"]} for item in ranked[:3]],
        "justificativa": " ".join(rationale) if rationale else "Sugestao baseada nas trilhas e cursos mais especificos atribuidos ao colaborador.",
    }


def empty_points_frame():
    return pd.DataFrame(
        columns=[
            "aluno",
            "aluno_key",
            "curso",
            "curso_key",
            "trilha_evento",
            "trilha_key",
            "tipo",
            "pontos",
            "dt",
            "ym",
            "is_manual",
            "is_training",
            "is_lesson",
            "is_completed",
            "is_approved",
            "milestone_value",
        ]
    )


def empty_lessons_frame():
    return pd.DataFrame(
        columns=[
            "aluno",
            "aluno_key",
            "curso",
            "curso_key",
            "aula",
            "aula_key",
            "dt",
            "ym",
            "duration_seconds",
        ]
    )


def load_products_map():
    path = Path("produtos_provas.csv")
    if not path.exists():
        print("Aviso: produtos_provas.csv nao encontrado. Cursos com prova serao inferidos pelo snapshot atual.")
        return {}

    df = pd.read_csv(path)
    df.columns = [norm_text(col) for col in df.columns]
    course_col = find_col(df, ["curso", "nome_produto", "produto", "nome do produto"])
    proof_col = find_col(df, ["possui_prova", "tem_prova", "tem prova", "possui prova"])
    if not course_col or not proof_col:
        print("Aviso: produtos_provas.csv sem colunas reconheciveis. Cursos com prova serao inferidos pelo snapshot atual.")
        return {}

    tmp = df[[course_col, proof_col]].copy()
    tmp["course_norm"] = tmp[course_col].map(norm_match)
    tmp["has_prova"] = tmp[proof_col].apply(parse_flag)
    tmp = tmp[tmp["course_norm"] != ""]
    if tmp.empty:
        return {}
    return tmp.groupby("course_norm")["has_prova"].max().to_dict()


def course_has_proof(course_name, proof_map):
    return proof_map.get(norm_match(course_name), False)


def load_cargo_profiles():
    path = Path("cargo_catalog.json")
    if not path.exists():
        print("Aviso: cargo_catalog.json nao encontrado. Sugestao de cargo sera desativada.")
        return [], []

    raw = json.loads(path.read_text(encoding="utf-8"))
    profiles = prepare_cargo_profiles(raw)
    catalog = [{"cargo": profile["cargo"], "source": profile["source"]} for profile in profiles]
    return profiles, catalog


def canonicalize_progress(df, proof_map):
    df = df.copy()
    df.columns = [norm_text(col) for col in df.columns]

    student_col = find_col(df, ["nome do aluno", "aluno", "usuario", "nome"])
    course_col = find_col(df, ["curso", "produto", "formacao"])
    trail_col = find_col(df, ["trilha"])
    progress_col = find_col(df, ["progresso (%)", "progresso", "progress"])
    grade_col = find_col(df, ["nota prova", "nota da prova", "nota", "resultado da prova", "resultado"])

    if not student_col or not course_col:
        raise SystemExit("progresso.xlsx nao possui colunas de aluno/curso reconheciveis.")

    rows = []
    for _, raw in df.iterrows():
        aluno = norm_text(raw.get(student_col))
        curso = norm_text(raw.get(course_col))
        if not aluno or not curso:
            continue

        trilha = norm_text(raw.get(trail_col)) if trail_col else ""
        trilha = trilha if trilha else "(sem trilha)"
        progresso = sanitize_progress(raw.get(progress_col)) if progress_col else 0.0
        nota_raw = raw.get(grade_col) if grade_col else None
        nota_value = parse_grade_value(nota_raw)
        has_prova = course_has_proof(curso, proof_map) or not np.isnan(nota_value)

        if np.isnan(nota_value):
            nota_resumida = "Nao possui prova" if not has_prova else "Nao realizada"
        else:
            nota_resumida = str(int(round(nota_value)))

        rows.append(
            {
                "aluno": aluno,
                "aluno_key": norm_match(aluno),
                "curso": curso,
                "curso_key": norm_match(curso),
                "trilha": trilha,
                "trilha_key": norm_match(trilha),
                "progresso_atual": progresso,
                "nota_atual": nota_value,
                "nota_atual_texto": nota_resumida,
                "has_prova": bool(has_prova),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out = out.drop_duplicates(
        subset=[
            "aluno_key",
            "curso_key",
            "trilha_key",
            "progresso_atual",
            "nota_atual_texto",
            "has_prova",
        ],
        keep="last",
    ).reset_index(drop=True)

    for label_col, key_col in [("aluno", "aluno_key"), ("curso", "curso_key"), ("trilha", "trilha_key")]:
        label_map = out.groupby(key_col)[label_col].agg(pick_display_label).to_dict()
        out[label_col] = out[key_col].map(label_map)

    return out


def canonicalize_points(df):
    if df is None or df.empty:
        return empty_points_frame()

    df = df.copy()
    df.columns = [norm_text(col) for col in df.columns]

    student_col = find_col(df, ["aluno", "nome", "usuario"])
    course_col = find_col(df, ["curso", "produto", "formacao"])
    trail_col = find_col(df, ["trilha"])
    type_col = find_col(df, ["tipo", "evento"])
    points_col = find_col(df, ["pontos"])
    date_col = find_col(df, ["criado em", "data", "created at"])

    if not student_col or not type_col or not date_col:
        raise SystemExit("pontos.xlsx nao possui colunas essenciais reconheciveis.")

    rows = []
    for _, raw in df.iterrows():
        aluno = norm_text(raw.get(student_col))
        if not aluno:
            continue

        curso = norm_text(raw.get(course_col)) if course_col else ""
        trilha = norm_text(raw.get(trail_col)) if trail_col else ""
        tipo = norm_text(raw.get(type_col)).lower()
        dt = parse_datetime(raw.get(date_col))
        ym = None if pd.isna(dt) else f"{dt.year:04d}-{dt.month:02d}"
        is_manual = tipo == "manual"
        is_training = tipo in TRAINING_EVENT_TYPES
        rows.append(
            {
                "aluno": aluno,
                "aluno_key": norm_match(aluno),
                "curso": curso,
                "curso_key": norm_match(curso),
                "trilha_evento": trilha if trilha else "(sem trilha)",
                "trilha_key": norm_match(trilha if trilha else "(sem trilha)"),
                "tipo": tipo,
                "pontos": int(round(fnum(raw.get(points_col), 0.0))) if points_col else 0,
                "dt": dt,
                "ym": ym,
                "is_manual": is_manual,
                "is_training": is_training,
                "is_lesson": tipo == "aula_concluida",
                "is_completed": tipo == "curso_concluido",
                "is_approved": tipo == "prova_aprovada",
                "milestone_value": MILESTONE_EVENT_VALUE.get(tipo, 0),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return empty_points_frame()

    out = out[out["ym"].notna()].copy()
    if out.empty:
        return empty_points_frame()

    out = out.drop_duplicates(
        subset=["aluno_key", "curso_key", "trilha_key", "tipo", "dt", "pontos"],
        keep="last",
    ).reset_index(drop=True)

    label_specs = [
        ("aluno", "aluno_key"),
        ("curso", "curso_key"),
        ("trilha_evento", "trilha_key"),
    ]
    for label_col, key_col in label_specs:
        label_map = out.groupby(key_col)[label_col].agg(pick_display_label).to_dict()
        out[label_col] = out[key_col].map(label_map)

    return out


def canonicalize_lessons(df):
    if df is None or df.empty:
        return empty_lessons_frame()

    df = df.copy()
    df.columns = [norm_text(col) for col in df.columns]

    student_col = find_col(df, ["nome do aluno", "nome", "aluno", "usuario"])
    course_col = find_col(df, ["curso", "produto", "formacao"])
    lesson_col = find_col(df, ["nome aula", "aula", "nome da aula", "licao", "lição"])
    date_col = find_col(df, ["data aula", "data", "criado em", "created at"])
    duration_col = find_col(df, ["tempo aula", "duracao", "duração", "tempo"])

    if not student_col or not course_col or not lesson_col or not date_col:
        raise SystemExit("aulas_assistidas.xlsx nao possui colunas essenciais reconheciveis.")

    rows = []
    for _, raw in df.iterrows():
        aluno = norm_text(raw.get(student_col))
        curso = norm_text(raw.get(course_col))
        aula = norm_text(raw.get(lesson_col))
        dt = parse_datetime(raw.get(date_col))
        if not aluno or not curso or not aula or pd.isna(dt):
            continue

        rows.append(
            {
                "aluno": aluno,
                "aluno_key": norm_match(aluno),
                "curso": curso,
                "curso_key": norm_match(curso),
                "aula": aula,
                "aula_key": norm_match(aula),
                "dt": dt,
                "ym": f"{dt.year:04d}-{dt.month:02d}",
                "duration_seconds": parse_duration_seconds(raw.get(duration_col)) if duration_col else 0,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return empty_lessons_frame()

    out = out.drop_duplicates(subset=["aluno_key", "curso_key", "aula_key", "dt"], keep="first").reset_index(drop=True)

    for label_col, key_col in [("aluno", "aluno_key"), ("curso", "curso_key"), ("aula", "aula_key")]:
        label_map = out.groupby(key_col)[label_col].agg(pick_display_label).to_dict()
        out[label_col] = out[key_col].map(label_map)

    return out


def build_course_catalog(lessons_df, progress_df):
    if lessons_df.empty:
        return {}

    course_obs = (
        lessons_df.groupby("curso_key", as_index=False)
        .agg(
            course_label=("curso", pick_display_label),
            observed_lessons=("aula_key", "nunique"),
            observed_rows=("aula_key", "size"),
        )
        .sort_values("course_label", kind="stable")
    )

    seen_counts = (
        lessons_df.groupby(["aluno_key", "curso_key"], as_index=False)
        .agg(seen_lessons=("aula_key", "nunique"))
    )
    progress_base = (
        progress_df.groupby(["aluno_key", "curso_key"], as_index=False)
        .agg(
            course_label=("curso", pick_display_label),
            progress_current=("progresso_atual", "max"),
        )
    )
    progress_with_lessons = progress_base.merge(seen_counts, on=["aluno_key", "curso_key"], how="left")
    progress_with_lessons["seen_lessons"] = progress_with_lessons["seen_lessons"].fillna(0)
    progress_with_lessons = progress_with_lessons.merge(
        course_obs[["curso_key", "observed_lessons"]],
        on="curso_key",
        how="left",
    )

    candidates = progress_with_lessons[
        (progress_with_lessons["progress_current"] > 0) & (progress_with_lessons["seen_lessons"] > 0)
    ].copy()
    if len(candidates):
        candidates["implied_total"] = candidates["seen_lessons"] / (candidates["progress_current"] / 100.0)
        candidates = candidates.replace([np.inf, -np.inf], np.nan)
        candidates = candidates[candidates["implied_total"].between(1, 500, inclusive="both")]
    else:
        candidates = pd.DataFrame(columns=["curso_key", "implied_total"])

    implied_stats = (
        candidates.groupby("curso_key", as_index=False)
        .agg(
            implied_median=("implied_total", "median"),
            implied_p75=("implied_total", lambda values: values.quantile(0.75)),
            implied_samples=("implied_total", "size"),
        )
        if len(candidates)
        else pd.DataFrame(columns=["curso_key", "implied_median", "implied_p75", "implied_samples"])
    )

    merged = course_obs.merge(implied_stats, on="curso_key", how="left")
    catalog = {}
    for _, row in merged.iterrows():
        observed_lessons = int(row["observed_lessons"])
        implied_median = None if pd.isna(row.get("implied_median")) else float(row["implied_median"])
        estimated_total = observed_lessons
        total_source = "aulas_observadas"
        if implied_median:
            estimated_total = max(observed_lessons, int(round(implied_median)))
            if estimated_total > observed_lessons:
                total_source = "estimado_aulas_e_snapshot"

        catalog[row["curso_key"]] = {
            "course_label": row["course_label"],
            "observed_lessons": observed_lessons,
            "estimated_total_lessons": int(max(estimated_total, 0)),
            "total_source": total_source if estimated_total > 0 else "indisponivel",
            "implied_median": None if implied_median is None else round(implied_median, 2),
            "implied_samples": int(0 if pd.isna(row.get("implied_samples")) else row["implied_samples"]),
        }

    return catalog


def build_people_profiles(progress_df, cargo_profiles):
    people_payload = {}
    for _, row in progress_df.iterrows():
        aluno = row["aluno"]
        people_payload.setdefault(aluno, {"trilhas": [], "cursos": []})
        if row["trilha"] and row["trilha"] != "(sem trilha)":
            people_payload[aluno]["trilhas"].append(row["trilha"])
        if row["curso"]:
            people_payload[aluno]["cursos"].append(row["curso"])

    result = {}
    for aluno in sorted(people_payload):
        payload = people_payload[aluno]
        result[aluno] = infer_cargo_profile(payload["trilhas"], payload["cursos"], cargo_profiles)
    return result


def build_assignments(progress_df, people_profiles, course_catalog):
    grouped = (
        progress_df.groupby(["aluno_key", "curso_key"], as_index=False)
        .agg(
            aluno=("aluno", pick_display_label),
            curso=("curso", pick_display_label),
            trilhas=("trilha", lambda values: unique_preserve(sorted([norm_text(v) for v in values if norm_text(v)]))),
            progresso_atual=("progresso_atual", "max"),
            nota_atual=("nota_atual", "max"),
            nota_atual_texto=("nota_atual_texto", pick_display_label),
            has_prova=("has_prova", "max"),
            snapshot_started=("progresso_atual", lambda values: any(fnum(value, 0.0) > 0 for value in values)),
            snapshot_completed=("progresso_atual", lambda values: any(is_completed_progress(value) for value in values)),
        )
        .sort_values(["aluno", "curso"], kind="stable")
        .reset_index(drop=True)
    )

    assignments = []
    for index, row in grouped.iterrows():
        profile = people_profiles.get(row["aluno"], {})
        course_meta = course_catalog.get(row["curso_key"], {})
        nota_atual = None if np.isnan(row["nota_atual"]) else round(float(row["nota_atual"]), 2)
        assignments.append(
            {
                "id": int(index + 1),
                "aluno": row["aluno"],
                "aluno_key": row["aluno_key"],
                "curso": row["curso"],
                "curso_key": row["curso_key"],
                "trilhas": row["trilhas"],
                "trail_count": len(row["trilhas"]),
                "cargo": profile.get("cargo_sugerido") or "Cargo nao identificado",
                "cargo_confianca": profile.get("confianca") or "baixa",
                "progresso_atual": round(float(row["progresso_atual"]), 2),
                "nota_atual": nota_atual,
                "nota_atual_texto": row["nota_atual_texto"],
                "has_prova": bool(row["has_prova"] or nota_atual is not None),
                "snapshot_started": bool(row["snapshot_started"]),
                "snapshot_completed": bool(row["snapshot_completed"]),
                "course_total_lessons": int(course_meta.get("estimated_total_lessons", 0)),
                "course_total_lessons_source": course_meta.get("total_source", "indisponivel"),
                "observed_lessons_catalog": int(course_meta.get("observed_lessons", 0)),
                "lesson_history_available": False,
                "observed_lessons_current": 0,
                "points_history_available": False,
                "any_history_available": False,
                "history_gap": False,
                "history_gap_current_pp": 0.0,
                "exam_approved_current": bool(nota_atual is not None and nota_atual >= NOTA_APROVACAO),
                "exam_date_estimated": None,
                "exam_month_estimated": None,
                "exam_date_source": None,
            }
        )
    return assignments


def infer_exam_dates(assignments, points_df, lessons_df):
    exam_records = {}

    if not assignments:
        return exam_records

    assignment_lookup = {(row["aluno_key"], row["curso_key"]): row["id"] for row in assignments}

    proof_map = {}
    complete_map = {}
    if not points_df.empty:
        points = points_df[(points_df["curso_key"] != "") & (~points_df["is_manual"])].copy()
        points["assignment_id"] = points.apply(lambda row: assignment_lookup.get((row["aluno_key"], row["curso_key"])), axis=1)
        points = points[points["assignment_id"].notna()].copy()
        if len(points):
            points["assignment_id"] = points["assignment_id"].astype(int)
            proof_map = (
                points[points["tipo"] == "prova_aprovada"]
                .groupby("assignment_id")["dt"]
                .max()
                .to_dict()
            )
            complete_map = (
                points[points["tipo"] == "curso_concluido"]
                .groupby("assignment_id")["dt"]
                .max()
                .to_dict()
            )

    lesson_last_map = {}
    if not lessons_df.empty:
        lessons = lessons_df.copy()
        lessons["assignment_id"] = lessons.apply(lambda row: assignment_lookup.get((row["aluno_key"], row["curso_key"])), axis=1)
        lessons = lessons[lessons["assignment_id"].notna()].copy()
        if len(lessons):
            lessons["assignment_id"] = lessons["assignment_id"].astype(int)
            lesson_firsts = (
                lessons.sort_values("dt", kind="stable")
                .drop_duplicates(subset=["assignment_id", "aula_key"], keep="first")
            )
            lesson_last_map = lesson_firsts.groupby("assignment_id")["dt"].max().to_dict()

    for assignment in assignments:
        if assignment["nota_atual"] is None:
            continue

        assignment_id = assignment["id"]
        dt = None
        source = None
        if assignment_id in proof_map:
            dt = proof_map[assignment_id]
            source = "prova_aprovada"
        elif assignment_id in complete_map:
            dt = complete_map[assignment_id]
            source = "curso_concluido"
        elif assignment_id in lesson_last_map:
            dt = lesson_last_map[assignment_id]
            source = "ultima_aula"
        else:
            source = "indisponivel"

        ym = None if dt is None or pd.isna(dt) else f"{dt.year:04d}-{dt.month:02d}"
        exam_records[assignment_id] = {
            "grade": float(assignment["nota_atual"]),
            "approved": bool(assignment["nota_atual"] >= NOTA_APROVACAO),
            "dt": None if dt is None or pd.isna(dt) else dt,
            "ym": ym,
            "source": source,
        }

        assignment["exam_date_estimated"] = None if dt is None or pd.isna(dt) else dt.date().isoformat()
        assignment["exam_month_estimated"] = ym
        assignment["exam_date_source"] = source

    return exam_records


def build_monthly_activity(assignments, points_df, lessons_df, exam_records, months, current_month):
    assignment_lookup = {(row["aluno_key"], row["curso_key"]): row["id"] for row in assignments}
    assignment_by_id = {row["id"]: row for row in assignments}
    by_month = {month: [] for month in months}
    month_context = {}

    manual_points_rows = 0
    matched_points_rows = empty_points_frame()
    points_lookup = {}
    points_history_assignments = set()

    if not points_df.empty:
        matched_points_rows = points_df[(points_df["curso_key"] != "")].copy()
        matched_points_rows["assignment_id"] = matched_points_rows.apply(
            lambda row: assignment_lookup.get((row["aluno_key"], row["curso_key"])),
            axis=1,
        )
        matched_points_rows = matched_points_rows[matched_points_rows["assignment_id"].notna()].copy()
        if len(matched_points_rows):
            matched_points_rows["assignment_id"] = matched_points_rows["assignment_id"].astype(int)
            matched_points_rows["training_dt"] = matched_points_rows["dt"].where(matched_points_rows["is_training"])
            points_history_assignments = set(
                matched_points_rows.loc[matched_points_rows["is_training"], "assignment_id"].astype(int).unique().tolist()
            )

            monthly_points = (
                matched_points_rows.groupby(["assignment_id", "ym"], as_index=False)
                .agg(
                    points_month=("pontos", "sum"),
                    event_count=("tipo", "size"),
                    training_events=("is_training", "sum"),
                    completed_events=("is_completed", "sum"),
                    approved_events=("is_approved", "sum"),
                    milestone_month=("milestone_value", "max"),
                    last_training_ts=("training_dt", "max"),
                )
            )

            for row in monthly_points.to_dict(orient="records"):
                points_lookup[(int(row["assignment_id"]), row["ym"])] = {
                    "points_month": int(row["points_month"]),
                    "event_count": int(row["event_count"]),
                    "training_events": int(row["training_events"]),
                    "completed_events": int(row["completed_events"]),
                    "approved_events": int(row["approved_events"]),
                    "milestone_month": int(row["milestone_month"]),
                    "last_training_ts": row["last_training_ts"],
                }

        manual_points_rows = int(points_df["is_manual"].sum())

    lesson_lookup = {}
    lesson_history_assignments = set()
    lessons_seen_by_assignment = {}
    unmatched_lesson_rows = 0

    if not lessons_df.empty:
        lesson_rows = lessons_df.copy()
        lesson_rows["assignment_id"] = lesson_rows.apply(
            lambda row: assignment_lookup.get((row["aluno_key"], row["curso_key"])),
            axis=1,
        )
        unmatched_lesson_rows = int(lesson_rows["assignment_id"].isna().sum())
        lesson_rows = lesson_rows[lesson_rows["assignment_id"].notna()].copy()
        if len(lesson_rows):
            lesson_rows["assignment_id"] = lesson_rows["assignment_id"].astype(int)
            lesson_firsts = (
                lesson_rows.sort_values("dt", kind="stable")
                .drop_duplicates(subset=["assignment_id", "aula_key"], keep="first")
            )
            lesson_firsts["ym"] = lesson_firsts["dt"].dt.strftime("%Y-%m")
            lesson_history_assignments = set(lesson_firsts["assignment_id"].astype(int).unique().tolist())
            lessons_seen_by_assignment = (
                lesson_firsts.groupby("assignment_id")["aula_key"].nunique().astype(int).to_dict()
            )

            monthly_lessons = (
                lesson_firsts.groupby(["assignment_id", "ym"], as_index=False)
                .agg(
                    lessons_month=("aula_key", "nunique"),
                    lesson_minutes_month=("duration_seconds", lambda values: round(float(values.sum()) / 60.0, 1)),
                    last_lesson_ts=("dt", "max"),
                )
            )
            for row in monthly_lessons.to_dict(orient="records"):
                lesson_lookup[(int(row["assignment_id"]), row["ym"])] = {
                    "lessons_month": int(row["lessons_month"]),
                    "lesson_minutes_month": float(row["lesson_minutes_month"]),
                    "last_lesson_ts": row["last_lesson_ts"],
                }

    any_history_assignments = set(points_history_assignments).union(lesson_history_assignments)

    for assignment in assignments:
        assignment_id = assignment["id"]
        assignment["lesson_history_available"] = assignment_id in lesson_history_assignments
        assignment["observed_lessons_current"] = int(lessons_seen_by_assignment.get(assignment_id, 0))
        assignment["points_history_available"] = assignment_id in points_history_assignments
        assignment["any_history_available"] = assignment_id in any_history_assignments
        assignment["history_gap"] = bool(assignment["progresso_atual"] > 0 and not assignment["any_history_available"])

        unique_lessons_until = 0
        milestone_until = 0
        observed_progress_prev = 0.0
        last_training_ts = None
        last_training_month = None
        approved_until = False

        for month in months:
            point_current = points_lookup.get((assignment_id, month), {})
            lesson_current = lesson_lookup.get((assignment_id, month), {})
            exam_meta = exam_records.get(assignment_id, {})

            lessons_month = int(lesson_current.get("lessons_month", 0))
            lesson_minutes_month = float(lesson_current.get("lesson_minutes_month", 0.0))
            unique_lessons_until += lessons_month

            if assignment["course_total_lessons"] > 0:
                lesson_progress_until = round(
                    clamp((unique_lessons_until / assignment["course_total_lessons"]) * 100.0, 0.0, 100.0),
                    2,
                )
            else:
                lesson_progress_until = 0.0

            milestone_until = max(milestone_until, int(point_current.get("milestone_month", 0)))
            observed_progress = max(lesson_progress_until, float(milestone_until))

            exam_taken_in_month = bool(exam_meta and exam_meta.get("ym") == month)
            exam_grade_month = float(exam_meta["grade"]) if exam_taken_in_month else None
            exam_approved_in_month = bool(exam_taken_in_month and exam_meta.get("approved"))

            completed_by_event = bool(point_current.get("completed_events", 0) > 0)
            approved_by_event = bool(point_current.get("approved_events", 0) > 0)
            approved_in_month = bool(exam_approved_in_month or approved_by_event)

            if completed_by_event or approved_in_month:
                observed_progress = 100.0

            completed_by_progress = bool(observed_progress_prev < COMPLETION_PROGRESS and observed_progress >= COMPLETION_PROGRESS)
            completed_in_month = bool(completed_by_event or completed_by_progress or approved_in_month)
            if completed_in_month:
                observed_progress = 100.0

            observed_progress = round(clamp(observed_progress, 0.0, 100.0), 2)
            progress_delta_month = round(max(0.0, observed_progress - observed_progress_prev), 2)
            started_in_month = bool(observed_progress_prev <= 0 and observed_progress > 0)

            effective_progress = observed_progress
            if month == current_month:
                effective_progress = round(max(effective_progress, assignment["progresso_atual"]), 2)

            started_until_month = bool(observed_progress > 0 or (month == current_month and assignment["snapshot_started"]))
            completed_until_month = bool(
                observed_progress >= COMPLETION_PROGRESS or (month == current_month and assignment["snapshot_completed"])
            )
            if completed_until_month:
                effective_progress = round(max(effective_progress, 100.0), 2)

            approved_until = bool(approved_until or approved_in_month)
            if month == current_month and exam_meta and exam_meta.get("approved") and not exam_meta.get("ym"):
                approved_until = True
            approved_until_month = bool(approved_until)

            candidate_dts = []
            if point_current.get("last_training_ts") is not None and not pd.isna(point_current.get("last_training_ts")):
                candidate_dts.append(point_current["last_training_ts"])
            if lesson_current.get("last_lesson_ts") is not None and not pd.isna(lesson_current.get("last_lesson_ts")):
                candidate_dts.append(lesson_current["last_lesson_ts"])
            if exam_taken_in_month and exam_meta.get("dt") is not None:
                candidate_dts.append(exam_meta["dt"])

            if candidate_dts:
                month_last_training_ts = max(candidate_dts)
                if last_training_ts is None or month_last_training_ts > last_training_ts:
                    last_training_ts = month_last_training_ts
                    last_training_month = month

            points_training_events = int(point_current.get("training_events", 0))
            event_count = int(point_current.get("event_count", 0))
            activity_events_month = points_training_events + lessons_month + (1 if exam_taken_in_month else 0)
            active_in_month = bool(lessons_month > 0 or points_training_events > 0 or exam_taken_in_month or progress_delta_month > 0)

            by_month[month].append(
                {
                    "assignment_id": assignment_id,
                    "points_month": int(point_current.get("points_month", 0)),
                    "event_count": int(event_count + lessons_month),
                    "training_events": int(activity_events_month),
                    "training_events_points": points_training_events,
                    "lessons_month": lessons_month,
                    "lesson_minutes_month": lesson_minutes_month,
                    "unique_lessons_until_month": int(unique_lessons_until),
                    "completed_count": 1 if completed_in_month else 0,
                    "approved_count": 1 if approved_in_month else 0,
                    "exam_taken_count": 1 if exam_taken_in_month else 0,
                    "active_in_month": active_in_month,
                    "started_in_month": started_in_month,
                    "completed_in_month": completed_in_month,
                    "approved_in_month": approved_in_month,
                    "started_until_month": started_until_month,
                    "completed_until_month": completed_until_month,
                    "approved_until_month": approved_until_month,
                    "progress_delta_month": progress_delta_month,
                    "progress_observed_until_month": observed_progress,
                    "progress_effective_until_month": effective_progress,
                    "lesson_progress_until_month": lesson_progress_until,
                    "milestone_until_month": int(milestone_until),
                    "exam_taken_in_month": exam_taken_in_month,
                    "exam_grade_month": exam_grade_month,
                    "exam_grade_source": exam_meta.get("source") if exam_taken_in_month else None,
                    "last_training_date": None if last_training_ts is None else last_training_ts.date().isoformat(),
                    "last_training_month": last_training_month,
                    "snapshot_progress_current": assignment["progresso_atual"],
                    "snapshot_grade_current": assignment["nota_atual"],
                    "history_gap": assignment["history_gap"],
                }
            )

            observed_progress_prev = observed_progress

        current_rows = [row for row in by_month[current_month] if row["assignment_id"] == assignment_id]
        current_observed = current_rows[0]["progress_observed_until_month"] if current_rows else 0.0
        assignment["history_gap_current_pp"] = round(max(0.0, assignment["progresso_atual"] - current_observed), 2)

    manual_df = points_df[points_df["is_manual"]].copy() if not points_df.empty else empty_points_frame()
    for month in months:
        month_rows = by_month[month]
        active_assignment_ids = [row["assignment_id"] for row in month_rows if row["active_in_month"]]
        active_collaborators = {assignment_by_id[row_id]["aluno"] for row_id in active_assignment_ids}
        progress_course_keys = {
            assignment_by_id[row["assignment_id"]]["curso_key"]
            for row in month_rows
            if row["progress_delta_month"] > 0
        }
        completed_course_keys = {
            assignment_by_id[row["assignment_id"]]["curso_key"]
            for row in month_rows
            if row["completed_in_month"]
        }
        approved_course_keys = {
            assignment_by_id[row["assignment_id"]]["curso_key"]
            for row in month_rows
            if row["approved_in_month"]
        }
        exam_course_scores = {}
        for row in month_rows:
            if not row["exam_taken_in_month"] or row["exam_grade_month"] is None:
                continue
            course_key = assignment_by_id[row["assignment_id"]]["curso_key"]
            exam_course_scores.setdefault(course_key, []).append(float(row["exam_grade_month"]))

        exam_scores = [float(np.mean(scores)) for scores in exam_course_scores.values()]
        month_context[month] = {
            "is_current_month": False,
            "has_any_data": bool(
                any(
                    row["active_in_month"]
                    or row["exam_taken_in_month"]
                    or row["completed_in_month"]
                    or row["approved_in_month"]
                    for row in month_rows
                )
            ),
            "active_assignments": int(len(active_assignment_ids)),
            "active_collaborators": int(len(active_collaborators)),
            "lessons_month": int(sum(row["lessons_month"] for row in month_rows)),
            "completed_courses_month": int(len(completed_course_keys)),
            "approved_exams_month": int(len(approved_course_keys)),
            "exam_takers_month": int(len(exam_course_scores)),
            "avg_exam_score_month": None if not exam_scores else round(float(np.mean(exam_scores)), 2),
            "avg_progress_observed_base": 0.0 if not month_rows else round(float(np.mean([row["progress_observed_until_month"] for row in month_rows])), 2),
            "avg_progress_effective_base": 0.0 if not month_rows else round(float(np.mean([row["progress_effective_until_month"] for row in month_rows])), 2),
            "courses_with_progress_month": int(len(progress_course_keys)),
            "progress_gain_month": round(float(sum(row["progress_delta_month"] for row in month_rows)), 2),
            "manual_points_rows": int(len(manual_df[manual_df["ym"] == month])) if len(manual_df) else 0,
        }

    activity_stats = {
        "points_history_assignments": int(len(points_history_assignments)),
        "lesson_history_assignments": int(len(lesson_history_assignments)),
        "any_history_assignments": int(len(any_history_assignments)),
        "unmatched_lesson_rows": int(unmatched_lesson_rows),
        "manual_points_rows": int(manual_points_rows),
    }
    return by_month, month_context, activity_stats


def main():
    proof_map = load_products_map()
    cargo_profiles, cargo_catalog = load_cargo_profiles()

    progresso_df = pd.read_excel("progresso.xlsx")
    pontos_df = pd.read_excel("pontos.xlsx")

    aulas_path = Path("aulas_assistidas.xlsx")
    if aulas_path.exists():
        aulas_df = pd.read_excel(aulas_path)
    else:
        print("Aviso: aulas_assistidas.xlsx nao encontrado. O calculo mensal voltara a depender apenas de pontos.xlsx.")
        aulas_df = pd.DataFrame()

    progress = canonicalize_progress(progresso_df, proof_map)
    points = canonicalize_points(pontos_df)
    lessons = canonicalize_lessons(aulas_df)

    if progress.empty:
        raise SystemExit("Sem dados validos em progresso.xlsx.")
    if points.empty and lessons.empty:
        raise SystemExit("Sem dados validos em pontos.xlsx nem em aulas_assistidas.xlsx.")

    current_date = datetime.now().astimezone().date()
    current_month = f"{current_date.year:04d}-{current_date.month:02d}"

    course_catalog = build_course_catalog(lessons, progress)
    people_profiles = build_people_profiles(progress, cargo_profiles)
    assignments = build_assignments(progress, people_profiles, course_catalog)
    exam_records = infer_exam_dates(assignments, points, lessons)

    months_with_data = set()
    if not points.empty:
        months_with_data.update(points["ym"].dropna().unique().tolist())
    if not lessons.empty:
        months_with_data.update(lessons["ym"].dropna().unique().tolist())
    months_with_data.update([record["ym"] for record in exam_records.values() if record.get("ym")])

    last_month_with_data = max(months_with_data) if months_with_data else current_month
    months = sorted(set(months_with_data).union({current_month}))

    monthly_activity, month_context, activity_stats = build_monthly_activity(
        assignments=assignments,
        points_df=points,
        lessons_df=lessons,
        exam_records=exam_records,
        months=months,
        current_month=current_month,
    )

    for month in months:
        month_context[month]["is_current_month"] = month == current_month

    history_gap_assignments = int(sum(1 for row in assignments if row["history_gap"]))
    collaborators_count = len({row["aluno"] for row in assignments})
    repeated_course_assignments = int(sum(1 for row in assignments if row["trail_count"] > 1))
    trails_catalog = sorted({trail for row in assignments for trail in row["trilhas"]})
    courses_catalog = sorted({row["curso"] for row in assignments})
    current_completed_assignments = int(sum(1 for row in assignments if row["snapshot_completed"]))
    open_assignments_current = int(sum(1 for row in assignments if not row["snapshot_completed"]))

    exam_numeric_count = int(sum(1 for row in assignments if row["nota_atual"] is not None))
    exam_dated_count = int(sum(1 for row in assignments if row["exam_month_estimated"]))
    courses_with_estimated_lesson_total = int(
        sum(1 for meta in course_catalog.values() if meta.get("total_source") == "estimado_aulas_e_snapshot")
    )

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "meses": months,
        "context": {
            "today": current_date.isoformat(),
            "current_month": current_month,
            "last_month_with_data": last_month_with_data,
            "has_current_month_data": bool(month_context.get(current_month, {}).get("has_any_data")),
            "month_context": month_context,
        },
        "methodology": {
            "assignment_source": "Snapshot atual de progresso.xlsx",
            "activity_source": "Primeiras aulas concluidas em aulas_assistidas.xlsx + eventos mensais de pontos.xlsx",
            "exam_date_rule": "prova_aprovada > curso_concluido > ultima aula assistida",
            "monthly_progress_observed_available": bool(not lessons.empty),
            "repeated_course_rule": "O mesmo curso e tratado uma unica vez por colaborador e vale para todas as trilhas em que ele aparece.",
            "limitations": [
                "progresso.xlsx continua sendo um snapshot atual e nao oferece historico por linha; por isso o mes atual pode usar o snapshot apenas como estado corrente complementar",
                "aulas_assistidas.xlsx permite reconstruir progresso observado por aula, mas alguns cursos ainda exigem estimativa de catalogo total de aulas a partir do snapshot atual",
                "a nota da prova e a nota final atual; quando nao existe evento proprio da prova, a data mensal e estimada por curso_concluido ou pela ultima aula assistida",
            ],
            "history_gap_assignments": history_gap_assignments,
            "repeated_course_assignments": repeated_course_assignments,
            "coverage": {
                "assignments_with_lesson_history": int(sum(1 for row in assignments if row["lesson_history_available"])),
                "assignments_with_points_history": int(sum(1 for row in assignments if row["points_history_available"])),
                "assignments_with_any_history": int(sum(1 for row in assignments if row["any_history_available"])),
                "exam_scores_numeric": exam_numeric_count,
                "exam_scores_dated": exam_dated_count,
                "courses_with_estimated_lesson_total": courses_with_estimated_lesson_total,
                "unmatched_lesson_rows": activity_stats["unmatched_lesson_rows"],
                "manual_points_rows": activity_stats["manual_points_rows"],
            },
            "source_rows": {
                "progresso_raw": int(len(progresso_df)),
                "progresso_modelado": int(len(progress)),
                "pontos_raw": int(len(pontos_df)),
                "pontos_modelado": int(len(points)),
                "aulas_assistidas_raw": int(len(aulas_df)),
                "aulas_assistidas_modelado": int(len(lessons)),
            },
        },
        "catalogs": {
            "trilhas": trails_catalog,
            "cursos": courses_catalog,
            "cargo_catalog": cargo_catalog,
        },
        "stats": {
            "assignments": len(assignments),
            "collaborators": collaborators_count,
            "unique_courses": len(courses_catalog),
            "current_completed_assignments": current_completed_assignments,
            "open_assignments_current": open_assignments_current,
        },
        "assignments": assignments,
        "activity_by_month": monthly_activity,
        "people_profiles": people_profiles,
    }

    with open("data_trilhas.json", "w", encoding="utf-8") as file:
        json.dump(out, file, ensure_ascii=False, indent=2)

    print("data_trilhas.json gerado com progresso mensal observado por aula e notas de prova datadas.")


if __name__ == "__main__":
    main()
