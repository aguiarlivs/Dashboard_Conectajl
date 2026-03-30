const ALL_TRAILS = "(Todas as trilhas)";
const STATUS_META = {
  concluiu_no_mes: { label:"Concluiu no mes", className:"ok" },
  em_desenvolvimento: { label:"Em desenvolvimento", className:"info" },
  sem_atividade: { label:"Sem atividade no mes", className:"warn" },
  nao_iniciou: { label:"Nao iniciou", className:"bad" }
};
const PRIORITY_META = {
  alta: { label:"Alta", className:"bad" },
  media: { label:"Media", className:"warn" },
  baixa: { label:"Baixa", className:"low" }
};

let DATA = null;
let STATE = null;
let charts = { status:null, progress:null, exam:null };

const $ = selector => document.querySelector(selector);

function escapeHtml(value){
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function formatInteger(value){
  return Number(value || 0).toLocaleString("pt-BR");
}

function formatDecimal(value, digits = 1){
  const num = Number(value);
  if(!Number.isFinite(num)) return "-";
  return num.toLocaleString("pt-BR", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits
  });
}

function formatPct(value, digits = 1){
  const num = Number(value);
  return Number.isFinite(num) ? `${formatDecimal(num, digits)}%` : "-";
}

function formatPp(value, digits = 1){
  const num = Number(value);
  return Number.isFinite(num) ? `${formatDecimal(num, digits)} pp` : "-";
}

function formatScore(value, digits = 1){
  const num = Number(value);
  return Number.isFinite(num) ? formatDecimal(num, digits) : "-";
}

function formatMinutes(value){
  const num = Number(value);
  return Number.isFinite(num) ? `${formatInteger(Math.round(num))} min` : "0 min";
}

function formatMonthLabel(ym){
  if(!ym || !String(ym).includes("-")) return ym || "-";
  const [year, month] = String(ym).split("-");
  const dt = new Date(Number(year), Number(month) - 1, 1);
  return dt.toLocaleDateString("pt-BR", { month:"long", year:"numeric" });
}

function formatDateLabel(isoDate){
  if(!isoDate) return "-";
  const dt = new Date(`${isoDate}T00:00:00`);
  return Number.isNaN(dt.getTime()) ? isoDate : dt.toLocaleDateString("pt-BR");
}

function summarizeList(values, limit = 2){
  const clean = [...new Set((values || []).filter(Boolean))];
  if(!clean.length) return "-";
  const visible = clean.slice(0, limit);
  const extra = clean.length - visible.length;
  return `${visible.join(", ")}${extra > 0 ? ` +${extra}` : ""}`;
}

function average(values){
  const clean = (values || [])
    .map(value => Number(value))
    .filter(value => Number.isFinite(value));
  if(!clean.length) return null;
  return clean.reduce((sum, value) => sum + value, 0) / clean.length;
}

function badge(label, className){
  return `<span class="badge ${className}">${escapeHtml(label)}</span>`;
}

function wrapCell(value){
  return `<span class="wrap">${escapeHtml(value ?? "-")}</span>`;
}

function buildState(data){
  const assignmentById = new Map(data.assignments.map(item => [item.id, item]));
  const activitiesByMonth = {};

  data.meses.forEach(month => {
    const monthRows = data.activity_by_month?.[month] || [];
    activitiesByMonth[month] = new Map(monthRows.map(item => [item.assignment_id, item]));
  });

  return { assignmentById, activitiesByMonth };
}

function defaultActivity(assignmentId){
  return {
    assignment_id: assignmentId,
    points_month: 0,
    event_count: 0,
    training_events: 0,
    training_events_points: 0,
    lessons_month: 0,
    lesson_minutes_month: 0,
    unique_lessons_until_month: 0,
    completed_count: 0,
    approved_count: 0,
    exam_taken_count: 0,
    active_in_month: false,
    started_in_month: false,
    completed_in_month: false,
    approved_in_month: false,
    started_until_month: false,
    completed_until_month: false,
    approved_until_month: false,
    progress_delta_month: 0,
    progress_observed_until_month: 0,
    progress_effective_until_month: 0,
    lesson_progress_until_month: 0,
    milestone_until_month: 0,
    exam_taken_in_month: false,
    exam_grade_month: null,
    exam_grade_source: null,
    last_training_date: null,
    last_training_month: null,
    snapshot_progress_current: 0,
    snapshot_grade_current: null,
    history_gap: false
  };
}

function populateFilters(){
  const trailSelect = $("#trailSelect");
  const monthSelect = $("#monthSelect");

  trailSelect.innerHTML = [ALL_TRAILS, ...(DATA.catalogs?.trilhas || [])]
    .map(item => `<option value="${escapeHtml(item)}">${escapeHtml(item)}</option>`)
    .join("");

  monthSelect.innerHTML = DATA.meses
    .map(month => `<option value="${month}">${escapeHtml(formatMonthLabel(month))}</option>`)
    .join("");

  trailSelect.value = ALL_TRAILS;
  monthSelect.value = DATA.context?.current_month || DATA.meses.at(-1) || "";
}

function getSelectedTrail(){
  return $("#trailSelect").value || ALL_TRAILS;
}

function getSelectedMonth(){
  return $("#monthSelect").value || DATA.context?.current_month || DATA.meses.at(-1) || "";
}

function getFilteredAssignments(trail){
  return DATA.assignments.filter(item => trail === ALL_TRAILS ? true : item.trilhas.includes(trail));
}

function buildAssignmentRows(month, trail){
  const activityMap = STATE.activitiesByMonth[month] || new Map();
  return getFilteredAssignments(trail).map(assignment => {
    const activity = activityMap.get(assignment.id) || defaultActivity(assignment.id);
    return { ...assignment, ...activity };
  });
}

function getProgressForMonthRow(row, month){
  return Number(month === DATA.context?.current_month ? row.progress_effective_until_month : row.progress_observed_until_month) || 0;
}

function computePriority(statusKey, progressAverage, notStartedCourses){
  if(statusKey === "nao_iniciou") return "alta";
  if(statusKey === "sem_atividade" && (progressAverage < 45 || notStartedCourses >= 2)) return "alta";
  if(statusKey === "sem_atividade") return "media";
  if(statusKey === "em_desenvolvimento" && progressAverage < 50) return "media";
  return "baixa";
}

function buildCollaboratorRows(assignmentRows, month){
  const groups = new Map();

  assignmentRows.forEach(row => {
    if(!groups.has(row.aluno)){
      groups.set(row.aluno, {
        aluno: row.aluno,
        cargo: row.cargo,
        trails: new Set(),
        assignedCourses: 0,
        activeCoursesMonth: 0,
        progressCoursesMonth: 0,
        progressGainMonth: 0,
        progressSum: 0,
        progressObservedSum: 0,
        completedCoursesMonth: 0,
        approvedExamsMonth: 0,
        examsTakenMonth: 0,
        lessonsMonth: 0,
        lessonMinutesMonth: 0,
        startedCourses: 0,
        completedCourses: 0,
        historyGapCourses: 0,
        activeCourseNames: [],
        lastTrainingMonth: null,
        examGradesMonth: [],
        examGradeTotalMonth: 0
      });
    }

    const item = groups.get(row.aluno);
    row.trilhas.forEach(trilha => item.trails.add(trilha));
    item.assignedCourses += 1;
    item.progressSum += getProgressForMonthRow(row, month);
    item.progressObservedSum += Number(row.progress_observed_until_month) || 0;
    if(row.active_in_month) item.activeCoursesMonth += 1;
    if(Number(row.progress_delta_month) > 0) item.progressCoursesMonth += 1;
    item.progressGainMonth += Number(row.progress_delta_month) || 0;
    item.completedCoursesMonth += Number(row.completed_count) || 0;
    item.approvedExamsMonth += Number(row.approved_count) || 0;
    item.examsTakenMonth += Number(row.exam_taken_count) || 0;
    item.lessonsMonth += Number(row.lessons_month) || 0;
    item.lessonMinutesMonth += Number(row.lesson_minutes_month) || 0;
    if(row.started_until_month) item.startedCourses += 1;
    if(row.completed_until_month) item.completedCourses += 1;
    if(row.history_gap) item.historyGapCourses += 1;
    if(row.active_in_month) item.activeCourseNames.push(row.curso);
    if(row.exam_taken_in_month){
      const gradeValue = Number.isFinite(Number(row.nota_atual)) ? Number(row.nota_atual) : Number(row.exam_grade_month);
      if(Number.isFinite(gradeValue)){
        item.examGradesMonth.push(gradeValue);
        item.examGradeTotalMonth += gradeValue;
      }
    }
    if(row.last_training_month && (!item.lastTrainingMonth || row.last_training_month > item.lastTrainingMonth)){
      item.lastTrainingMonth = row.last_training_month;
    }
  });

  return [...groups.values()].map(item => {
    const trailsList = [...item.trails].sort((a, b) => a.localeCompare(b, "pt-BR"));
    const progressAverage = item.assignedCourses ? item.progressSum / item.assignedCourses : 0;
    const progressObservedAverage = item.assignedCourses ? item.progressObservedSum / item.assignedCourses : 0;
    const notStartedCourses = Math.max(0, item.assignedCourses - item.startedCourses);
    let statusKey = "nao_iniciou";

    if(item.completedCoursesMonth > 0 || item.approvedExamsMonth > 0){
      statusKey = "concluiu_no_mes";
    } else if(item.activeCoursesMonth > 0 || item.progressCoursesMonth > 0){
      statusKey = "em_desenvolvimento";
    } else if(item.startedCourses > 0 || item.historyGapCourses > 0){
      statusKey = "sem_atividade";
    }

    const priorityKey = computePriority(statusKey, progressAverage, notStartedCourses);
    return {
      ...item,
      trailsLabel: `${formatInteger(trailsList.length)} trilha${trailsList.length === 1 ? "" : "s"} • ${summarizeList(trailsList, 2)}`,
      activeCourseNamesLabel: item.activeCourseNames.length ? summarizeList(item.activeCourseNames, 3) : "-",
      lastTrainingMonthLabel: item.lastTrainingMonth ? formatMonthLabel(item.lastTrainingMonth) : "Sem historico no periodo",
      progressAverage,
      progressObservedAverage,
      examAverageMonth: average(item.examGradesMonth),
      examGradeTotalMonth: item.examGradeTotalMonth,
      examGradeCountMonth: item.examGradesMonth.length,
      notStartedCourses,
      openCourses: Math.max(0, item.assignedCourses - item.completedCourses),
      statusKey,
      statusLabel: STATUS_META[statusKey].label,
      priorityKey,
      priorityLabel: PRIORITY_META[priorityKey].label
    };
  });
}

function buildCourseRows(assignmentRows, month, selectedTrail){
  const groups = new Map();

  assignmentRows.forEach(row => {
    const key = row.curso;
    if(!groups.has(key)){
      groups.set(key, {
        curso: row.curso,
        trails: new Set(),
        assignedCollaborators: new Set(),
        activeCollaborators: new Set(),
        completedCollaborators: new Set(),
        approvedCollaborators: new Set(),
        examsTakenMonth: 0,
        examGradesMonth: [],
        lessonsMonth: 0,
        lessonMinutesMonth: 0,
        progressGainMonth: 0,
        progressAssignmentsMonth: 0,
        progressSum: 0,
        startedAssignments: 0,
        completedAssignments: 0
      });
    }

    const item = groups.get(key);
    row.trilhas.forEach(trilha => item.trails.add(trilha));
    item.assignedCollaborators.add(row.aluno);
    item.progressSum += getProgressForMonthRow(row, month);
    item.progressGainMonth += Number(row.progress_delta_month) || 0;
    if(Number(row.progress_delta_month) > 0) item.progressAssignmentsMonth += 1;
    if(row.active_in_month) item.activeCollaborators.add(row.aluno);
    if(row.completed_in_month) item.completedCollaborators.add(row.aluno);
    if(row.approved_in_month) item.approvedCollaborators.add(row.aluno);
    item.examsTakenMonth += Number(row.exam_taken_count) || 0;
    if(Number.isFinite(Number(row.exam_grade_month))) item.examGradesMonth.push(Number(row.exam_grade_month));
    item.lessonsMonth += Number(row.lessons_month) || 0;
    item.lessonMinutesMonth += Number(row.lesson_minutes_month) || 0;
    if(row.started_until_month) item.startedAssignments += 1;
    if(row.completed_until_month) item.completedAssignments += 1;
  });

  return [...groups.values()].map(item => {
    const assigned = item.assignedCollaborators.size;
    const active = item.activeCollaborators.size;
    const completed = item.completedCollaborators.size;
    const approved = item.approvedCollaborators.size;
    let statusKey = "nao_iniciou";

    if(completed > 0 || approved > 0){
      statusKey = "concluiu_no_mes";
    } else if(active > 0 || item.progressAssignmentsMonth > 0){
      statusKey = "em_desenvolvimento";
    } else if(item.startedAssignments > 0){
      statusKey = "sem_atividade";
    }

    const trailsList = [...item.trails].sort((a, b) => a.localeCompare(b, "pt-BR"));
    return {
      curso: item.curso,
      trailsLabel: selectedTrail === ALL_TRAILS ? summarizeList(trailsList, 2) : selectedTrail,
      assignedCollaborators: assigned,
      activeCollaborators: active,
      completedCollaborators: completed,
      approvedCollaborators: approved,
      examsTakenMonth: item.examsTakenMonth,
      examAverageMonth: average(item.examGradesMonth),
      lessonsMonth: item.lessonsMonth,
      lessonMinutesMonth: item.lessonMinutesMonth,
      progressAverage: assigned ? item.progressSum / assigned : 0,
      progressGainMonth: item.progressGainMonth,
      progressAssignmentsMonth: item.progressAssignmentsMonth,
      startedAssignments: item.startedAssignments,
      completedAssignments: item.completedAssignments,
      statusKey,
      statusLabel: STATUS_META[statusKey].label
    };
  });
}

function summarize(collaboratorRows, assignmentRows, courseRows, month, selectedTrail){
  const activeCollaborators = collaboratorRows.filter(item => ["concluiu_no_mes", "em_desenvolvimento"].includes(item.statusKey)).length;
  const inactiveCollaborators = collaboratorRows.length - activeCollaborators;
  const monthMeta = DATA.context?.month_context?.[month] || {};
  const useGlobalMonthMeta = selectedTrail === ALL_TRAILS;
  const examCourseRows = courseRows.filter(item => item.examsTakenMonth > 0);
  const localAvgExamScoreMonth = average(examCourseRows.map(item => item.examAverageMonth));
  const localExamsTakenMonth = examCourseRows.length;
  const localCompletedCoursesMonth = courseRows.filter(item => item.completedCollaborators > 0).length;
  const localApprovedExamsMonth = courseRows.filter(item => item.approvedCollaborators > 0).length;
  const localProgressCoursesMonth = courseRows.filter(item => item.progressAssignmentsMonth > 0).length;

  const progressAverage = average(assignmentRows.map(item => getProgressForMonthRow(item, month))) || 0;
  const progressObservedAverage = average(assignmentRows.map(item => Number(item.progress_observed_until_month) || 0)) || 0;
  const progressGainMonth = assignmentRows.reduce((sum, item) => sum + (Number(item.progress_delta_month) || 0), 0);
  const lessonsMonth = assignmentRows.reduce((sum, item) => sum + (Number(item.lessons_month) || 0), 0);
  const lessonMinutesMonth = assignmentRows.reduce((sum, item) => sum + (Number(item.lesson_minutes_month) || 0), 0);
  const historyGapAssignments = assignmentRows.filter(item => item.history_gap).length;

  return {
    collaborators: collaboratorRows.length,
    assignments: assignmentRows.length,
    uniqueCourses: courseRows.length,
    activeCollaborators,
    inactiveCollaborators,
    activeRate: collaboratorRows.length ? (activeCollaborators / collaboratorRows.length) * 100 : 0,
    inactiveRate: collaboratorRows.length ? (inactiveCollaborators / collaboratorRows.length) * 100 : 0,
    progressAverage,
    progressObservedAverage,
    progressCoursesMonth: useGlobalMonthMeta ? Number(monthMeta.courses_with_progress_month || 0) : localProgressCoursesMonth,
    progressGainMonth,
    completedCoursesMonth: useGlobalMonthMeta ? Number(monthMeta.completed_courses_month || 0) : localCompletedCoursesMonth,
    examsTakenMonth: useGlobalMonthMeta ? Number(monthMeta.exam_takers_month || 0) : localExamsTakenMonth,
    approvedExamsMonth: useGlobalMonthMeta ? Number(monthMeta.approved_exams_month || 0) : localApprovedExamsMonth,
    avgExamScoreMonth: useGlobalMonthMeta && monthMeta.avg_exam_score_month != null
      ? Number(monthMeta.avg_exam_score_month)
      : localAvgExamScoreMonth,
    lessonsMonth,
    lessonMinutesMonth,
    historyGapAssignments
  };
}

function filterRows(rows, query, fields){
  const term = (query || "").trim().toLowerCase();
  if(!term) return rows;
  return rows.filter(row => fields.some(field => String(row[field] ?? "").toLowerCase().includes(term)));
}

function renderTable(targetSelector, rows, columns){
  const target = $(targetSelector);
  if(!rows.length){
    target.innerHTML = '<div class="empty">Sem registros para o filtro atual.</div>';
    return;
  }

  const head = `<thead><tr>${columns.map(col => {
    const alignClass = col.align ? ` align-${col.align}` : "";
    return `<th class="${alignClass.trim()}">${escapeHtml(col.label)}</th>`;
  }).join("")}</tr></thead>`;
  const body = `<tbody>${rows.map(row => `<tr>${columns.map(col => {
    const value = row[col.key];
    const content = col.render ? col.render(value, row) : escapeHtml(value ?? "-");
    const alignClass = col.align ? ` align-${col.align}` : "";
    return `<td class="${alignClass.trim()}">${content}</td>`;
  }).join("")}</tr>`).join("")}</tbody>`;

  target.innerHTML = `<table>${head}${body}</table>`;
}

function destroyChart(name){
  if(charts[name]){
    charts[name].destroy();
    charts[name] = null;
  }
}

function chartBaseOptions(){
  return {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode:"index", intersect:false },
    plugins: {
      legend: {
        labels: {
          color:"#4f5a64",
          usePointStyle: true,
          boxWidth: 10
        }
      },
      tooltip: {
        callbacks: {
          label(context){
            const label = context.dataset?.label || "";
            const value = context.parsed?.y;
            if(context.dataset?.yAxisID === "y1"){
              return `${label}: ${formatPct(value, 1)}`;
            }
            return `${label}: ${formatInteger(value)}`;
          }
        }
      }
    }
  };
}

function renderStatusChart(collaboratorRows){
  destroyChart("status");
  const ctx = document.getElementById("statusChart").getContext("2d");
  const buckets = [
    { key:"concluiu_no_mes", label:"Concluiu no mes", value:collaboratorRows.filter(item => item.statusKey === "concluiu_no_mes").length, color:"#1f8a5b" },
    { key:"em_desenvolvimento", label:"Em desenvolvimento", value:collaboratorRows.filter(item => item.statusKey === "em_desenvolvimento").length, color:"#123c61" },
    { key:"sem_atividade", label:"Sem atividade", value:collaboratorRows.filter(item => item.statusKey === "sem_atividade").length, color:"#a76817" },
    { key:"nao_iniciou", label:"Nao iniciou", value:collaboratorRows.filter(item => item.statusKey === "nao_iniciou").length, color:"#b34535" }
  ];

  charts.status = new Chart(ctx, {
    type: "bar",
    data: {
      labels: buckets.map(item => item.label),
      datasets: [{
        data: buckets.map(item => item.value),
        backgroundColor: buckets.map(item => item.color),
        borderRadius: 12,
        borderSkipped: false
      }]
    },
    options: {
      ...chartBaseOptions(),
      plugins: { legend: { display:false } },
      scales: {
        x: {
          ticks: { color:"#4f5a64" },
          grid: { display:false }
        },
        y: {
          beginAtZero: true,
          ticks: { color:"#4f5a64", precision:0 },
          grid: { color:"rgba(23,33,43,.08)" }
        }
      }
    }
  });
}

function buildTrendSeries(selectedTrail){
  const labels = [];
  const progressAverageSeries = [];
  const progressCoursesSeries = [];
  const examsTakenSeries = [];
  const examAverageSeries = [];

  DATA.meses.forEach(month => {
    const assignments = buildAssignmentRows(month, selectedTrail);
    const collaboratorRows = buildCollaboratorRows(assignments, month);
    const courseRows = buildCourseRows(assignments, month, selectedTrail);
    const summary = summarize(collaboratorRows, assignments, courseRows, month, selectedTrail);

    labels.push(formatMonthLabel(month));
    progressAverageSeries.push(summary.progressObservedAverage);
    progressCoursesSeries.push(summary.progressCoursesMonth);
    examsTakenSeries.push(summary.examsTakenMonth);
    examAverageSeries.push(summary.avgExamScoreMonth);
  });

  return { labels, progressAverageSeries, progressCoursesSeries, examsTakenSeries, examAverageSeries };
}

function renderProgressChart(selectedTrail){
  destroyChart("progress");
  const ctx = document.getElementById("progressChart").getContext("2d");
  const series = buildTrendSeries(selectedTrail);

  charts.progress = new Chart(ctx, {
    data: {
      labels: series.labels,
      datasets: [
        {
          type: "bar",
          label: "Cursos com avancos",
          data: series.progressCoursesSeries,
          backgroundColor: "rgba(197,106,44,.72)",
          borderRadius: 10,
          yAxisID: "y"
        },
        {
          type: "line",
          label: "Progresso medio observado",
          data: series.progressAverageSeries,
          borderColor: "#123c61",
          backgroundColor: "rgba(18,60,97,.12)",
          tension: 0.28,
          fill: true,
          pointRadius: 3,
          pointHoverRadius: 4,
          yAxisID: "y1"
        }
      ]
    },
    options: {
      ...chartBaseOptions(),
      scales: {
        x: {
          ticks: { color:"#4f5a64", maxRotation:0, autoSkip:true },
          grid: { display:false }
        },
        y: {
          beginAtZero: true,
          ticks: { color:"#4f5a64", precision:0 },
          grid: { color:"rgba(23,33,43,.08)" }
        },
        y1: {
          beginAtZero: true,
          max: 100,
          position: "right",
          ticks: {
            color:"#123c61",
            callback: value => `${value}%`
          },
          grid: { drawOnChartArea:false }
        }
      }
    }
  });
}

function renderExamChart(selectedTrail){
  destroyChart("exam");
  const ctx = document.getElementById("examChart").getContext("2d");
  const series = buildTrendSeries(selectedTrail);

  charts.exam = new Chart(ctx, {
    data: {
      labels: series.labels,
      datasets: [
        {
          type: "bar",
          label: "Nota media",
          data: series.examAverageSeries,
          backgroundColor: "rgba(167,104,23,.68)",
          borderRadius: 10,
          yAxisID: "y1"
        },
        {
          type: "line",
          label: "Provas distintas no mes",
          data: series.examsTakenSeries,
          borderColor: "#1f8a5b",
          backgroundColor: "rgba(31,138,91,.15)",
          tension: 0.24,
          spanGaps: true,
          pointRadius: 3,
          pointHoverRadius: 4,
          yAxisID: "y"
        }
      ]
    },
    options: {
      ...chartBaseOptions(),
      scales: {
        x: {
          ticks: { color:"#4f5a64", maxRotation:0, autoSkip:true },
          grid: { display:false }
        },
        y: {
          beginAtZero: true,
          ticks: { color:"#4f5a64", precision:0 },
          grid: { color:"rgba(23,33,43,.08)" }
        },
        y1: {
          beginAtZero: true,
          max: 100,
          position: "right",
          ticks: { color:"#a76817" },
          grid: { drawOnChartArea:false }
        }
      }
    }
  });
}

function renderSummary(summary, month){
  $("#kpiEligible").textContent = formatInteger(summary.collaborators);
  $("#kpiEligibleMeta").textContent = `${formatInteger(summary.uniqueCourses)} cursos distintos no recorte`;

  $("#kpiActive").textContent = formatInteger(summary.activeCollaborators);
  $("#kpiActiveMeta").textContent = `${formatPct(summary.activeRate)} da base elegivel`;

  $("#kpiProgressAvg").textContent = formatPct(summary.progressAverage);
  $("#kpiProgressAvgMeta").textContent = month === DATA.context?.current_month
    ? "Fotografia atual do snapshot no mes selecionado"
    : "Progresso observado ate o fim do mes";

  $("#kpiProgressCourses").textContent = formatInteger(summary.progressCoursesMonth);
  $("#kpiProgressCoursesMeta").textContent = summary.progressCoursesMonth
    ? `${formatPp(summary.progressGainMonth)} distribuidos no mes`
    : "Sem ganho de progresso datado no mes";

  $("#kpiCompleted").textContent = formatInteger(summary.completedCoursesMonth);
  $("#kpiCompletedMeta").textContent = "Cursos distintos que chegaram a 100%";

  $("#kpiExams").textContent = formatInteger(summary.examsTakenMonth);
  $("#kpiExamsMeta").textContent = `${formatInteger(summary.approvedExamsMonth)} aprovacao(oes) distinta(s) no mes`;

  $("#kpiScore").textContent = summary.avgExamScoreMonth == null ? "-" : formatScore(summary.avgExamScoreMonth, 1);
  $("#kpiScoreMeta").textContent = summary.examsTakenMonth
    ? `Base de ${formatInteger(summary.examsTakenMonth)} prova(s) distinta(s)`
    : "Sem prova datada no mes";

  $("#kpiLessons").textContent = formatInteger(summary.lessonsMonth);
  $("#kpiLessonsMeta").textContent = `${formatMinutes(summary.lessonMinutesMonth)} assistidos no mes`;
}

function renderStatusBar(summary, month, selectedTrail){
  const context = DATA.context || {};
  const monthMeta = context.month_context?.[month] || {};
  const trailLabel = selectedTrail === ALL_TRAILS ? "Todas as trilhas" : selectedTrail;

  let text = `${formatInteger(summary.activeCollaborators)} de ${formatInteger(summary.collaborators)} colaboradores tiveram atividade datada no mes.`;

  if(month === context.current_month && !monthMeta.has_any_data){
    text = `${formatMonthLabel(month)} ainda nao possui atividade datada ate ${formatDateLabel(context.today)}. O progresso medio usa a fotografia atual do snapshot para representar o estado corrente.`;
  } else if(!monthMeta.has_any_data){
    text = `${formatMonthLabel(month)} nao possui atividade datada no recorte atual. O painel preserva a fotografia acumulada ate o fim do mes selecionado.`;
  } else {
    text = `${formatInteger(summary.progressCoursesMonth)} curso(s) distintos avancaram no mes, ${formatInteger(summary.examsTakenMonth)} prova(s) distinta(s) receberam data e o progresso medio observado da base fechou em ${formatPct(summary.progressObservedAverage)}.`;
  }

  $("#statusTitle").textContent = `${formatMonthLabel(month)} • ${trailLabel}`;
  $("#statusText").textContent = text;
}

function renderActiveTable(collaboratorRows){
  const query = $("#activeSearch").value;
  const rows = filterRows(
    collaboratorRows
      .filter(item => ["concluiu_no_mes", "em_desenvolvimento"].includes(item.statusKey))
      .sort((a, b) =>
        b.progressGainMonth - a.progressGainMonth ||
        b.completedCoursesMonth - a.completedCoursesMonth ||
        b.lessonsMonth - a.lessonsMonth ||
        a.aluno.localeCompare(b.aluno, "pt-BR")
      ),
    query,
    ["aluno", "cargo", "trailsLabel", "activeCourseNamesLabel", "statusLabel"]
  );

  $("#activeSubtitle").textContent = rows.length
    ? `${formatInteger(rows.length)} colaborador(es) com avancos datados ou conclusoes no mes.`
    : "Nenhum colaborador com movimento datado no mes.";

  renderTable("#activeTable", rows, [
    { key:"statusLabel", label:"Status", align:"center", render:(value, row) => badge(value, STATUS_META[row.statusKey].className) },
    { key:"aluno", label:"Colaborador" },
    { key:"cargo", label:"Cargo" },
    { key:"trailsLabel", label:"Trilhas", render:value => wrapCell(value) },
    { key:"assignedCourses", label:"Cursos com acesso", align:"center", render:value => formatInteger(value) },
    { key:"activeCoursesMonth", label:"Cursos ativos no mes", align:"center", render:value => formatInteger(value) },
    { key:"progressAverage", label:"Progresso medio", align:"center", render:value => formatPct(value, 1) },
    { key:"lessonsMonth", label:"Aulas", align:"center", render:value => formatInteger(value) },
    { key:"completedCoursesMonth", label:"Concluidos", align:"center", render:value => formatInteger(value) },
    {
      key:"examGradeTotalMonth",
      label:"Notas prova no mes",
      align:"center",
      render:(value, row) => row.examGradeCountMonth
        ? wrapCell(row.examGradeCountMonth > 1 ? `${formatScore(value, 0)} (${formatInteger(row.examGradeCountMonth)} provas)` : formatScore(value, 0))
        : '<span class="muted">-</span>'
    },
    { key:"activeCourseNamesLabel", label:"Cursos com movimento", render:value => wrapCell(value) }
  ]);
}

function renderInactiveTable(collaboratorRows){
  const query = $("#inactiveSearch").value;
  const priorityOrder = ["alta", "media", "baixa"];
  const rows = filterRows(
    collaboratorRows
      .filter(item => ["sem_atividade", "nao_iniciou"].includes(item.statusKey))
      .sort((a, b) => {
        const priorityDelta = priorityOrder.indexOf(a.priorityKey) - priorityOrder.indexOf(b.priorityKey);
        return priorityDelta ||
          b.openCourses - a.openCourses ||
          a.lastTrainingMonthLabel.localeCompare(b.lastTrainingMonthLabel, "pt-BR") ||
          a.aluno.localeCompare(b.aluno, "pt-BR");
      }),
    query,
    ["aluno", "cargo", "trailsLabel", "statusLabel", "priorityLabel", "lastTrainingMonthLabel"]
  );

  $("#inactiveSubtitle").textContent = rows.length
    ? `${formatInteger(rows.length)} colaborador(es) sem atividade datada no mes selecionado.`
    : "Toda a base elegivel teve alguma movimentacao datada no mes.";

  renderTable("#inactiveTable", rows, [
    { key:"priorityLabel", label:"Prioridade", align:"center", render:(value, row) => badge(value, PRIORITY_META[row.priorityKey].className) },
    { key:"statusLabel", label:"Status", align:"center", render:(value, row) => badge(value, STATUS_META[row.statusKey].className) },
    { key:"aluno", label:"Colaborador" },
    { key:"cargo", label:"Cargo" },
    { key:"trailsLabel", label:"Trilhas", render:value => wrapCell(value) },
    { key:"progressAverage", label:"Progresso medio", align:"center", render:value => formatPct(value, 1) },
    { key:"startedCourses", label:"Ja iniciados", align:"center", render:value => formatInteger(value) },
    { key:"notStartedCourses", label:"Sem iniciar", align:"center", render:value => formatInteger(value) },
    { key:"lastTrainingMonthLabel", label:"Ultimo mes com atividade", render:value => wrapCell(value) }
  ]);
}

function renderCourseTable(courseRows){
  const query = $("#courseSearch").value;
  const rows = filterRows(
    courseRows.sort((a, b) =>
      b.progressGainMonth - a.progressGainMonth ||
      b.activeCollaborators - a.activeCollaborators ||
      b.completedCollaborators - a.completedCollaborators ||
      a.curso.localeCompare(b.curso, "pt-BR")
    ),
    query,
    ["curso", "trailsLabel", "statusLabel"]
  );

  $("#courseSubtitle").textContent = rows.length
    ? `${formatInteger(rows.length)} curso(s) no recorte atual.`
    : "Nenhum curso no recorte atual.";

  renderTable("#courseTable", rows, [
    { key:"statusLabel", label:"Status", align:"center", render:(value, row) => badge(value, STATUS_META[row.statusKey].className) },
    { key:"curso", label:"Curso", render:value => wrapCell(value) },
    { key:"trailsLabel", label:"Trilhas", render:value => wrapCell(value) },
    { key:"assignedCollaborators", label:"Colaboradores com acesso", align:"center", render:value => formatInteger(value) },
    { key:"activeCollaborators", label:"Ativos no mes", align:"center", render:value => formatInteger(value) },
    { key:"progressAverage", label:"Progresso medio", align:"center", render:value => formatPct(value, 1) },
    { key:"lessonsMonth", label:"Aulas", align:"center", render:value => formatInteger(value) },
    { key:"examsTakenMonth", label:"Provas no mes", align:"center", render:value => formatInteger(value) },
    { key:"examAverageMonth", label:"Nota media", align:"center", render:value => formatScore(value, 1) },
    { key:"completedCollaborators", label:"Concluiu no mes", align:"center", render:value => formatInteger(value) }
  ]);
}

function render(){
  const month = getSelectedMonth();
  const selectedTrail = getSelectedTrail();
  const assignmentRows = buildAssignmentRows(month, selectedTrail);
  const collaboratorRows = buildCollaboratorRows(assignmentRows, month);
  const courseRows = buildCourseRows(assignmentRows, month, selectedTrail);
  const summary = summarize(collaboratorRows, assignmentRows, courseRows, month, selectedTrail);

  renderStatusBar(summary, month, selectedTrail);
  renderSummary(summary, month);
  renderStatusChart(collaboratorRows);
  renderProgressChart(selectedTrail);
  renderExamChart(selectedTrail);
  renderActiveTable(collaboratorRows);
  renderInactiveTable(collaboratorRows);
  renderCourseTable(courseRows);
}

function wireEvents(){
  $("#trailSelect").addEventListener("change", render);
  $("#monthSelect").addEventListener("change", render);
  $("#activeSearch").addEventListener("input", () => renderActiveTable(buildCollaboratorRows(buildAssignmentRows(getSelectedMonth(), getSelectedTrail()), getSelectedMonth())));
  $("#inactiveSearch").addEventListener("input", () => renderInactiveTable(buildCollaboratorRows(buildAssignmentRows(getSelectedMonth(), getSelectedTrail()), getSelectedMonth())));
  $("#courseSearch").addEventListener("input", () => renderCourseTable(buildCourseRows(buildAssignmentRows(getSelectedMonth(), getSelectedTrail()), getSelectedMonth(), getSelectedTrail())));
}

async function loadData(){
  const response = await fetch("./data_trilhas.json", { cache:"no-store" });
  if(!response.ok) throw new Error("Falha ao carregar data_trilhas.json");
  DATA = await response.json();
  STATE = buildState(DATA);
  populateFilters();
  wireEvents();
  render();
}

document.addEventListener("DOMContentLoaded", () => {
  loadData().catch(error => {
    alert(`Falha ao carregar o dashboard: ${error.message}`);
  });
});
