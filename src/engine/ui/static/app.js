/* engine · live mode — the page. Vanilla JS, no build.
   Everything shown comes from the JSON views the server serves (engine.report.views)
   and from the job API; this file only formats. Text is set with textContent, never
   innerHTML, so a value from a run directory is never parsed as markup.

   The one-click path: the Setup panel says what was found (GET /api/setup), the New
   run form lists the case files (GET /api/cases) and "Create and run" makes the run
   and starts the six stages as one pipeline job (POST /api/runs/<run>/pipeline),
   whose `stage` events drive the 01–06 strip in the job panel. */
(function () {
  'use strict';

  var $ = function (id) { return document.getElementById(id); };
  var STAGES = [['01_ingest', 'ingest'], ['02_retrieve', 'retrieve'], ['03_filter', 'filter'],
                ['04_rank', 'rank'], ['05_reason', 'reason'], ['06_medicine', 'medicine']];
  var TABS = [['report', 'Report'], ['candidates', 'Candidates'], ['ranking', 'Ranking'], ['chain', 'Evidence chain'],
              ['medicine', 'Medicine'], ['provenance', 'Provenance']];
  var CITE = /\[([a-z][a-z0-9-]*:[^\]\s]+)\]/g;

  var state = { runs: [], run: null, summary: null, tab: 'report', stages: null, providers: null, setup: null, cases: [],
                setupOpen: null, job: null, es: null, jobStage: null, setupJob: null, setupEs: null, opts: {} };

  // ------------------------------------------------------------------ helpers

  function h(tag, attrs) {
    var el = document.createElement(tag);
    if (attrs) {
      Object.keys(attrs).forEach(function (k) {
        var v = attrs[k];
        if (v === null || v === undefined || v === false) { return; }
        if (k === 'class') { el.className = v; }
        else if (k === 'text') { el.textContent = v; }
        else if (k === 'hidden') { el.hidden = !!v; }
        else if (k === 'checked' || k === 'disabled' || k === 'selected') { el[k] = !!v; }
        else if (k.slice(0, 2) === 'on') { el.addEventListener(k.slice(2), v); }
        else { el.setAttribute(k, v); }
      });
    }
    for (var i = 2; i < arguments.length; i++) { append(el, arguments[i]); }
    return el;
  }
  function append(el, child) {
    if (child === null || child === undefined || child === false) { return; }
    if (Array.isArray(child)) { child.forEach(function (c) { append(el, c); }); return; }
    if (typeof child === 'string' || typeof child === 'number') { el.appendChild(document.createTextNode(String(child))); return; }
    el.appendChild(child);
  }
  function clear(el) { while (el.firstChild) { el.removeChild(el.firstChild); } return el; }
  function dash(v) { return (v === null || v === undefined || v === '') ? '—' : String(v); }
  function num(v) {
    if (v === null || v === undefined || v === '') { return h('span', { class: 'num faint', text: '—' }); }
    var text = (typeof v === 'number' && Number.isInteger(v)) ? v.toLocaleString('en-US') : String(v);
    return h('span', { class: 'num', text: text });
  }
  function af(v) {
    if (v === null || v === undefined || v === '') { return num(v); }
    var s = String(v);
    if (s.length > 10 && /^[0-9.eE+-]+$/.test(s)) {
      var n = Number(s);
      var short = isFinite(n) ? (n === 0 ? '0' : n.toExponential(2)) : s;
      return h('span', { class: 'num', title: s, text: short });
    }
    return h('span', { class: 'num', text: s });
  }
  function when(iso) {
    if (!iso) { return '—'; }
    return String(iso).replace('T', ' ').replace(/\+00:00$/, ' UTC').replace(/Z$/, ' UTC');
  }
  function secs(v) { return (v === null || v === undefined) ? '—' : (Number(v).toLocaleString('en-US', { maximumFractionDigits: 1 }) + ' s'); }
  function gib(bytes) { return (bytes === null || bytes === undefined) ? '—' : (Number(bytes) / 1073741824).toLocaleString('en-US', { maximumFractionDigits: 1 }) + ' GiB'; }
  function mark(text, kind) { return h('span', { class: 'mark' + (kind ? ' ' + kind : ''), text: text }); }
  function label(v) { return v === null || v === undefined ? '—' : String(v).replace(/_/g, ' '); }
  function classificationMark(v) {
    if (!v) { return mark('no classification', ''); }
    var kind = /^(pathogenic|likely_pathogenic)$/.test(v) ? 'crit' : /benign/.test(v) ? 'good' : 'warn';
    return mark(label(v), kind);
  }
  function idLink(entry) {
    if (!entry) { return null; }
    if (typeof entry === 'string') { entry = { id: entry, url: null }; }
    if (entry.url) { return h('a', { class: 'id', href: entry.url, rel: 'noopener', target: '_blank', text: entry.id }); }
    return h('span', { class: 'id unresolved', title: 'no record with this id in any evidence store of the run', text: entry.id });
  }
  function idLinks(entries, sep) {
    var out = [];
    (entries || []).forEach(function (e, i) { if (i) { out.push(sep === undefined ? ', ' : sep); } out.push(idLink(e)); });
    return out.length ? out : [h('span', { class: 'faint', text: '—' })];
  }
  function refMap(entries) {
    var m = {};
    (entries || []).forEach(function (e) { if (e && e.id) { m[e.id] = e; } });
    return m;
  }
  function cited(text, refs) {
    // [record:id] tokens become links when the id resolves in the view's references.
    var frag = document.createDocumentFragment(), s = String(text || ''), last = 0, m;
    CITE.lastIndex = 0;
    while ((m = CITE.exec(s)) !== null) {
      frag.appendChild(document.createTextNode(s.slice(last, m.index) + '['));
      frag.appendChild(idLink(refs && refs[m[1]] ? refs[m[1]] : { id: m[1], url: null }));
      frag.appendChild(document.createTextNode(']'));
      last = m.index + m[0].length;
    }
    frag.appendChild(document.createTextNode(s.slice(last)));
    return frag;
  }
  function table(headers, rows, cls) {
    var thead = h('thead', null, h('tr', null, headers.map(function (hd) {
      var isObj = hd && typeof hd === 'object';
      return h('th', { class: isObj && hd.num ? 'num' : null, text: isObj ? hd.text : hd });
    })));
    var tbody = h('tbody', null, rows.map(function (cells) {
      return h('tr', null, cells.map(function (c) {
        if (c && typeof c === 'object' && !(c instanceof Node) && !Array.isArray(c) && 'cell' in c) {
          return h('td', { class: c.class || null, title: c.title || null }, c.cell);
        }
        return h('td', null, c);
      }));
    }));
    return h('div', { class: 'tbl' }, h('table', { class: cls || null }, thead, tbody));
  }
  function facts(pairs) {
    var dl = h('dl', { class: 'facts' });
    pairs.forEach(function (p) {
      if (p[1] === undefined) { return; }
      dl.appendChild(h('dt', { text: p[0] }));
      dl.appendChild(h('dd', null, p[1] === null ? h('span', { class: 'faint', text: '—' }) : p[1]));
    });
    return dl;
  }
  function kv(obj) {
    var dl = h('dl', { class: 'kv' });
    Object.keys(obj || {}).forEach(function (k) {
      var v = obj[k];
      var val = (v !== null && typeof v === 'object') ? h('span', { class: 'mono small', text: JSON.stringify(v) }) : num(v);
      dl.appendChild(h('div', null, h('dt', { text: k }), h('dd', null, val)));
    });
    return dl;
  }
  function list(items, cls, refs) {
    if (!items || !items.length) { return h('p', { class: 'muted small', text: 'none' }); }
    return h('ul', { class: cls || null }, items.map(function (t) { return h('li', { class: 'wide' }, cited(t, refs)); }));
  }
  function notice(text, kind) { return h('p', { class: 'notice' + (kind ? ' ' + kind : ''), text: text }); }
  function pre(doc) { return h('pre', { text: JSON.stringify(doc, null, 1) }); }

  function api(path, opts) {
    opts = opts || {};
    var init = { method: opts.method || 'GET', headers: {} };
    if (opts.body !== undefined) { init.headers['Content-Type'] = 'application/json'; init.body = JSON.stringify(opts.body); }
    return fetch(path, init).then(function (r) {
      var ct = r.headers.get('Content-Type') || '';
      var p = ct.indexOf('json') >= 0 ? r.json() : r.text().then(function (t) { return { error: t }; });
      return p.then(function (doc) {
        if (!r.ok) { var e = new Error((doc && doc.error) || (r.status + ' ' + r.statusText)); e.status = r.status; e.doc = doc; throw e; }
        return doc;
      });
    });
  }

  // -------------------------------------------------------------------- theme

  (function theme() {
    var root = document.documentElement, btn = $('theme'), order = ['system', 'light', 'dark'], current = 'system';
    function read() { try { return localStorage.getItem('engine-ui-theme') || 'system'; } catch (e) { return 'system'; } }
    function save(v) { try { localStorage.setItem('engine-ui-theme', v); } catch (e) { /* private window */ } }
    function apply(v) {
      current = order.indexOf(v) >= 0 ? v : 'system';
      if (current === 'system') { root.removeAttribute('data-theme'); } else { root.setAttribute('data-theme', current); }
      btn.textContent = 'Theme: ' + current;
    }
    apply(read());
    btn.addEventListener('click', function () { apply(order[(order.indexOf(current) + 1) % order.length]); save(current); });
  })();

  // -------------------------------------------------------------------- setup

  function loadSetup() {
    return api('/api/setup').then(function (doc) {
      state.setup = doc;
      renderSetup();
      renderPipelineOpts();
      if (doc.setup_job && !isFinished(doc.setup_job) && !(state.setupJob && state.setupJob.id === doc.setup_job.id)) { startSetupJob(doc.setup_job); }
      return doc;
    }).catch(function (e) {
      state.setup = { items: [], error: e.message, pipeline: null };
      renderSetup();
    });
  }
  function loadCases() {
    return api('/api/cases').then(function (doc) { state.cases = doc.cases || []; renderCasePicker(); })
      .catch(function (e) { state.cases = []; $('case-info').textContent = 'Could not list case files: ' + e.message; });
  }

  function setupOpen() {
    if (state.setupOpen !== null) { return state.setupOpen; }
    return !state.runs.length;  // an empty work directory opens with the Setup panel; runs present keep it folded
  }
  function renderSetup() {
    var s = state.setup, box = $('setup'), items = clear($('setup-items')), open = setupOpen();
    box.className = 'setup' + (open ? '' : ' folded');
    $('setup-toggle').textContent = open ? 'Hide setup' : 'Setup';
    if (!s) { $('setup-summary').textContent = 'Loading…'; return; }
    if (s.error) { $('setup-summary').textContent = 'Could not read the setup: ' + s.error; return; }
    var ready = s.items.filter(function (i) { return i.ready; }).length;
    var missing = s.items.filter(function (i) { return !i.ready && i.setup_stage; });
    $('setup-summary').textContent = ready + ' of ' + s.items.length + ' ready' + (missing.length ? ' — missing: ' + missing.map(function (i) { return i.label.toLowerCase(); }).join(', ') : '');
    $('setup-project').textContent = 'project ' + s.project + ' · repo ' + s.repo;
    if (!open) { return; }
    var busy = state.setupJob && !isFinished(state.setupJob);
    s.items.forEach(function (i) {
      var li = h('li', { class: 'setup-item ' + i.state },
        h('span', { class: 'dot ' + i.state, title: i.state }),
        h('div', { class: 'body' },
          h('div', { class: 'row' }, h('span', { class: 'label', text: i.label }), ' ', mark(i.state, { ready: 'good', partial: 'warn', missing: '' }[i.state]),
            i.source !== 'discovered' && i.source !== 'default' ? h('span', { class: 'small muted', text: ' · from ' + (i.source === 'env' ? 'the environment' : 'an option') }) : null),
          h('div', { class: 'path mono small', text: i.path || 'not configured' }),
          h('p', { class: 'small muted detail', text: i.detail }),
          !i.ready && i.setup_stage ? h('div', { class: 'row act' },
            h('code', { class: 'small', text: i.command }),
            h('button', { class: 'btn small', type: 'button', disabled: !!busy, onclick: function () { runSetupStage(i.setup_stage); } }, busy ? 'busy' : 'Run ' + i.setup_stage)) : null));
      items.appendChild(li);
    });
    if (s.docker) {
      items.appendChild(h('li', { class: 'setup-item ' + (s.docker.fallback ? 'missing' : 'ready') },
        h('span', { class: 'dot ' + (s.docker.fallback ? 'missing' : 'ready') }),
        h('div', { class: 'body' },
          h('div', { class: 'row' }, h('span', { class: 'label', text: 'Docker' }), ' ', s.docker.fallback ? mark('not reachable', '') : mark('ready', 'good')),
          h('p', { class: 'small muted detail', text: s.docker.fallback
            ? 'docker info could not be read: rank would use the pin\'s fallback heap ' + s.docker.heap + ' and needs a running daemon'
            : 'daemon memory ' + gib(s.docker.docker_memory_bytes) + ' · rank heap ' + s.docker.heap + ' (memory minus the 1 GiB reserve)' }))));
    }
  }
  $('setup-toggle').addEventListener('click', function () { state.setupOpen = !setupOpen(); renderSetup(); });

  function runSetupStage(stage) {
    api('/api/setup/jobs', { method: 'POST', body: { stage: stage } }).then(function (job) { startSetupJob(job); })
      .catch(function (e) { $('setup-summary').textContent = 'Refused (' + (e.status || '') + '): ' + e.message; });
  }
  function startSetupJob(job) {
    if (state.setupEs) { state.setupEs.close(); state.setupEs = null; }
    state.setupJob = job;
    var box = $('setup-job'), log = clear($('setup-job-log'));
    box.hidden = false;
    renderSetupJobStatus();
    renderSetup();
    state.setupEs = stream(job, {
      status: function (d) { if (d.argv) { logInto(log, '$ ' + d.argv.join(' '), 'sys'); } if (d.status && d.status !== 'cancelling') { state.setupJob.status = d.status; } renderSetupJobStatus(); },
      line: function (d) { logInto(log, d.text, d.stream === 'stderr' ? 'err' : ''); },
      done: function (d) {
        state.setupJob.status = d.status; state.setupJob.exit_code = d.exit_code;
        logInto(log, 'exit ' + d.exit_code + ' · ' + d.status, d.status === 'done' ? 'sys' : 'err');
        state.setupEs = null;
        renderSetupJobStatus();
        loadSetup();
      }
    });
  }
  function renderSetupJobStatus() {
    var j = state.setupJob, el = clear($('setup-job-status'));
    if (!j) { return; }
    var kind = j.status === 'done' ? 'good' : j.status === 'failed' ? 'crit' : j.status === 'cancelled' ? 'warn' : '';
    el.appendChild(mark(j.status, kind));
    el.appendChild(h('span', null, ' ' + j.stage + ' · job ' + j.id + ' · started ' + when(j.started_at) + ' · ', h('a', { href: '/api/jobs/' + j.id + '/log', target: '_blank', rel: 'noopener', text: 'log file' })));
    $('setup-job-cancel').disabled = isFinished(j);
  }
  $('setup-job-cancel').addEventListener('click', function () {
    if (!state.setupJob) { return; }
    api('/api/jobs/' + state.setupJob.id + '/cancel', { method: 'POST' }).catch(function (e) { $('setup-summary').textContent = 'Cancel refused: ' + e.message; });
  });

  // --------------------------------------------------------- pipeline options

  function pipelineDefaults() { return (state.setup && state.setup.pipeline) || { live: false, any_key: false, provider: 'fake', model: 'fake-model', effort: 'high', top: 3, cost: null }; }

  function renderPipelineOpts() {
    ['new-run-pipeline', 'run-pipeline-opts'].forEach(function (id) { buildPipelineOpts($(id), id); });
  }
  function buildPipelineOpts(box, key) {
    var d = pipelineDefaults(), prior = state.opts[key] || {};
    var live = prior.live !== undefined ? prior.live : d.live, rerun = !!prior.rerun, top = prior.top || d.top;
    clear(box);
    var liveIn = h('input', { type: 'checkbox', checked: live, disabled: !d.any_key, id: key + '-live' });
    var rerunIn = h('input', { type: 'checkbox', checked: rerun, id: key + '-rerun' });
    var topIn = h('input', { type: 'number', class: 'mono', min: 1, max: 1000, step: 1, value: top, id: key + '-top' });
    var models = d.models && d.models.length ? d.models : [{ id: d.model, label: d.model }];
    var modelSel = h('select', { class: 'mono', id: key + '-model' });
    models.forEach(function (m) { modelSel.appendChild(h('option', { value: m.id, text: m.label + ' — ' + m.id })); });
    modelSel.appendChild(h('option', { value: '__other__', text: 'other OpenRouter id…' }));
    var modelOther = h('input', { type: 'text', class: 'mono', placeholder: 'author/slug', id: key + '-model-other', hidden: true });
    var effSel = h('select', { id: key + '-effort' });
    (d.efforts || ['low', 'medium', 'high', 'xhigh', 'max']).forEach(function (e) { effSel.appendChild(h('option', { value: e, text: e })); });
    modelSel.value = prior.model && models.some(function (m) { return m.id === prior.model; }) ? prior.model : (prior.model ? '__other__' : models[0].id);
    if (modelSel.value === '__other__') { modelOther.value = prior.model || ''; modelOther.hidden = false; }
    effSel.value = prior.effort || d.effort;
    var modeEl = h('p', { class: 'mode small wide' });
    function chosenModel() { return modelSel.value === '__other__' ? (modelOther.value || '').trim() : modelSel.value; }
    function sync() {
      var on = liveIn.checked;
      modelOther.hidden = modelSel.value !== '__other__';
      state.opts[key] = { live: on, rerun: rerunIn.checked, top: Number(topIn.value) || d.top, model: chosenModel(), effort: effSel.value };
      box.className = 'pipeline-opts' + (on ? ' live' : '');
      clear(modeEl);
      if (on) {
        modeEl.appendChild(h('span', { class: 'state', text: 'LIVE' }));
        modeEl.appendChild(h('span', null, ' — provider ', h('b', { text: d.provider }), ' · model ', h('span', { class: 'mono', text: chosenModel() || d.model }), ' · effort ', h('b', { text: effSel.value }),
          ' · cost band ', h('b', { text: d.cost ? d.cost.band : 'paid' }), d.cost && d.cost.note ? h('span', { class: 'muted', text: ': ' + costNote(d, Number(topIn.value) || d.top) }) : null));
      } else {
        modeEl.appendChild(h('span', { class: 'state', text: 'OFF' }));
        modeEl.appendChild(h('span', null, d.any_key
          ? ' — stages 5–6 run with the scripted fake provider: no model is called, nothing is paid for; the chain and the report are empty documents.'
          : ' — no model key in the server environment (set OPENROUTER_API_KEY or ANTHROPIC_API_KEY, or the .env above the checkout): stages 5–6 run with the scripted fake provider.'));
      }
    }
    liveIn.addEventListener('change', sync); rerunIn.addEventListener('change', sync); topIn.addEventListener('input', sync);
    modelSel.addEventListener('change', sync); modelOther.addEventListener('input', sync); effSel.addEventListener('change', sync);
    box.appendChild(h('label', { class: 'check' }, liveIn, h('span', null, h('b', { text: 'Live mode' }), h('span', { class: 'muted small', text: ' — call the model in stages 5 and 6' }))));
    box.appendChild(modeEl);
    box.appendChild(h('div', { class: 'row' },
      h('label', { class: 'check' }, rerunIn, h('span', null, 'Re-run all', h('span', { class: 'muted small', text: ' — do not skip stages whose outputs are fresh' }))),
      h('label', { class: 'check inline' }, h('span', { text: 'Top candidates' }), topIn),
      h('label', { class: 'check inline' }, h('span', { text: 'Model' }), modelSel, modelOther),
      h('label', { class: 'check inline' }, h('span', { text: 'Effort' }), effSel)));
    sync();
  }
  function costNote(d, top) {
    var calls = top + 1;
    return 'up to ' + calls + ' agent conversations (' + top + ' candidate' + (top === 1 ? '' : 's') + ' + the medicine report), each up to ' + (d.cost.max_turns || 10) + ' tool-calling turns at effort ' + d.effort + ', billed by ' + d.provider;
  }
  function readOpts(key) {
    var d = pipelineDefaults(), o = state.opts[key] || {};
    return { live: o.live !== undefined ? o.live : d.live, rerun: !!o.rerun, top: o.top || d.top, model: o.model || d.model, effort: o.effort || d.effort };
  }

  // --------------------------------------------------------------------- runs

  function loadRuns() {
    return api('/api/runs').then(function (doc) {
      state.runs = doc.runs;
      $('work').textContent = doc.work;
      renderRunList();
      return doc;
    }).catch(function (e) { $('run-list-msg').textContent = 'Could not list runs: ' + e.message; });
  }

  function stageState(st, job) {
    if (job && !isFinished(job) && (job.stage_dir === st.dir || job.current_dir === st.dir)) { return ['running', 'running']; }
    if (job && !isFinished(job) && job.pipeline) {
      var ps = job.pipeline.stages.filter(function (p) { return p.dir === st.dir; })[0];
      if (ps && ps.status === 'pending') { return ['queued', 'none']; }
    }
    if (st.failed && st.failed.length) { return ['failed on ' + st.failed.length + ' candidate' + (st.failed.length > 1 ? 's' : ''), 'failed']; }
    if (!st.present) { return ['not run', 'none']; }
    if (!st.manifest) { return ['no manifest', 'dry']; }  // the directory is there, the stage did not finish: shown like a dry run, not a failure
    if (st.dry_run) { return ['dry run', 'dry']; }
    return ['done', 'done'];
  }
  function isFinished(job) { return !job || /^(done|failed|cancelled)$/.test(job.status); }

  function renderRunList() {
    var ul = clear($('run-list'));
    if (!state.runs.length) { $('run-list-msg').textContent = 'No runs under the work directory yet.'; return; }
    $('run-list-msg').textContent = state.runs.length + ' run' + (state.runs.length > 1 ? 's' : '');
    state.runs.forEach(function (r) {
      var last = null;
      r.stages.forEach(function (s) { if (s.finished_at && (!last || s.finished_at > last)) { last = s.finished_at; } });
      var strip = h('div', { class: 'strip', title: r.stages_present.join(', ') || 'no stage yet' },
        r.stages.map(function (s) { return h('span', { class: stageState(s, r.job)[1], title: s.dir + ': ' + stageState(s, r.job)[0] }); }));
      var running = r.job && !isFinished(r.job) ? 'running ' + (r.job.kind === 'pipeline' ? 'pipeline' + (r.job.current_stage ? ' · ' + r.job.current_stage : '') : r.job.stage) + ' · ' : '';
      var btn = h('button', { type: 'button', onclick: function () { selectRun(r.name); } },
        h('div', { class: 'name', text: r.name }), strip,
        h('div', { class: 'when', text: running + (last ? 'last stage ' + when(last) : 'no stage finished') }));
      ul.appendChild(h('li', { class: r.name === state.run ? 'current' : null }, btn));
    });
  }

  function selectRun(name, tab) {
    state.run = name;
    if (tab) { state.tab = tab; }
    location.hash = '#run=' + encodeURIComponent(name) + '&tab=' + state.tab;
    renderRunList();
    $('empty').hidden = true;
    $('run').hidden = false;
    renderSetup();
    return loadSummary().then(function () { renderTabs(); renderTab(); attachRunningJob(); });
  }

  function loadSummary() {
    return api('/api/runs/' + encodeURIComponent(state.run) + '/summary').then(function (s) {
      state.summary = s;
      renderRunHead(s);
      renderStages(s);
    }).catch(function (e) {
      clear($('run-title')).textContent = state.run;
      clear($('run-facts'));
      clear($('stages')).appendChild(h('li', { text: 'Could not read the run: ' + e.message }));
    });
  }

  function renderRunHead(s) {
    $('run-title').textContent = s.run_name + (s.sample ? ' · ' + s.sample : '');
    var vcf = s.vcf ? h('span', null, h('span', { class: 'mono small', text: s.vcf.path }), ' · ', num(s.vcf.bytes), ' bytes · sha256 ',
                                  h('span', { class: 'mono small', text: (s.vcf.sha256 || '').slice(0, 16) + '…', title: s.vcf.sha256 })) : null;
    var rec = s.record || {};
    clear($('run-facts')).appendChild(facts([
      ['Directory', h('span', { class: 'mono small', text: s.run_dir })],
      ['Sample', s.sample ? h('span', { class: 'mono', text: s.sample }) : h('span', { class: 'faint', text: 'stage 1 has not recorded one' })],
      ['Case file', rec.case_path ? h('span', null, h('span', { class: 'mono small', text: rec.case_path }), ' ', h('span', { class: 'faint small', text: '(' + rec.source + ')' })) : h('span', { class: 'faint', text: 'unknown — create runs from the page to record it' })],
      ['VCF', vcf],
      ['HPO terms', s.hpo && s.hpo.length ? h('span', null, s.hpo.map(function (t) { return h('span', { class: 'chip', text: t }); }), h('span', { class: 'faint small', text: ' from ' + s.hpo_source })) : h('span', { class: 'faint', text: 'none recorded yet (stage 4 or 5 records them)' })],
      ['Engine versions', s.engine_versions && s.engine_versions.length ? s.engine_versions.join(', ') : null],
      ['Report', h('a', { href: '/api/runs/' + encodeURIComponent(s.run_name) + '/report.html', target: '_blank', rel: 'noopener', text: 'open the rendered report in a new tab' })]
    ]));
  }

  function renderStages(s) {
    var ol = clear($('stages'));
    var job = state.job && state.job.run === s.run_name && !isFinished(state.job) ? state.job : null;
    s.stages.forEach(function (st) {
      var ss = stageState(st, job);
      var kind = { running: '', done: 'good', dry: 'warn', failed: 'crit', none: '' }[ss[1]];
      var li = h('li', { class: ss[1] === 'running' ? 'running' : null },
        h('div', { class: 'n', text: st.dir.slice(0, 2) }),
        h('div', { class: 'name', text: st.stage }),
        h('div', { class: 'status' }, mark(ss[0], kind)),
        st.manifest ? h('p', { class: 'meta', text: (st.finished_at ? when(st.finished_at) : '—') + (st.duration_s !== null && st.duration_s !== undefined ? ' · ' + secs(st.duration_s) : '') + (st.wall_time_s ? ' · wall ' + secs(st.wall_time_s) : '') }) : null,
        st.headline && st.headline.length ? h('p', { class: 'counts' }, st.headline.map(function (p, j) {
          return [j ? ' · ' : null, p[0] + ' ', h('b', { text: dash(p[1]) })];
        })) : null,
        h('div', { class: 'act' }, h('button', { class: 'btn small', type: 'button', disabled: !!job,
          onclick: function () { openJob(st.stage); } }, job ? 'busy' : 'Run ' + st.stage)));
      ol.appendChild(li);
    });
    $('pipeline-run').disabled = !!job;
  }

  // --------------------------------------------------------------------- tabs

  function renderTabs() {
    var nav = clear($('tabs'));
    TABS.forEach(function (t) {
      nav.appendChild(h('button', { type: 'button', class: t[0] === state.tab ? 'active' : null,
        onclick: function () { state.tab = t[0]; location.hash = '#run=' + encodeURIComponent(state.run) + '&tab=' + t[0]; renderTabs(); renderTab(); } }, t[1]));
    });
  }

  function renderTab() {
    var body = clear($('tab-body'));
    var run = encodeURIComponent(state.run), tab = state.tab;
    if (tab === 'report') { renderReport(run, body); return; }
    var path = '/api/runs/' + run + '/' + tab + (tab === 'ranking' ? '?top=20' : '');
    body.appendChild(h('p', { class: 'muted small', text: 'Loading ' + tab + '…' }));
    api(path).then(function (doc) {
      if (state.tab !== tab) { return; }
      clear(body);
      ({ candidates: renderCandidates, ranking: renderRanking, chain: renderChain, medicine: renderMedicine, provenance: renderProvenance })[tab](doc, body);
    }).catch(function (e) { clear(body).appendChild(notice('Could not load ' + tab + ': ' + e.message, 'crit')); });
  }

  // ------------------------------------------------------------------- report

  function renderReport(run, body) {
    var url = '/api/runs/' + run + '/report.html';
    var frame = h('iframe', { class: 'report', src: url, title: 'rendered report', loading: 'lazy' });
    frame.addEventListener('load', function () {
      try { frame.style.height = Math.max(600, frame.contentDocument.documentElement.scrollHeight + 32) + 'px'; } catch (e) { /* keep the default height */ }
    });
    body.appendChild(h('p', { class: 'small muted wide' }, 'The rendered report — the same views as one self-contained document. ',
      h('a', { href: url, target: '_blank', rel: 'noopener', text: 'Open it in a new tab' }), '.'));
    body.appendChild(frame);
  }

  // ---------------------------------------------------------------- candidates

  function renderCandidates(c, body) {
    body.appendChild(h('h3', { text: 'Candidates' }));
    if (!c.present) { body.appendChild(notice('Stage 3 has not run: no 03_filter/candidates.json in this run.')); return; }
    var counts = c.counts || {};
    body.appendChild(facts([
      ['Rows in · kept · dropped', h('span', null, num(counts.rows_in), ' · ', num(counts.kept), ' · ', num(counts.dropped))],
      ['Candidates', h('span', null, num(counts.candidates), counts.candidates_by_model ? h('span', { class: 'muted small', text: ' (' + Object.keys(counts.candidates_by_model).map(function (k) { return k + ' ' + counts.candidates_by_model[k]; }).join(', ') + ')' }) : null)],
      ['Blind ranker', c.rank_present
        ? h('span', null, c.top_agreement === true ? mark('top candidate is Exomiser rank 1', 'good') : mark('top candidate is not Exomiser rank 1', 'warn'), ' ',
                          c.order_agreement === true ? mark('order agrees', 'good') : c.order_agreement === false ? mark('order differs', 'warn') : mark('nothing to compare', ''))
        : h('span', { class: 'faint', text: 'stage 4 has not run' })]
    ]));
    if (c.join_check && c.join_check.stale) {
      body.appendChild(notice('The stage-4 join is older than the shortlist (missing: ' + (c.join_check.missing.join(', ') || 'none') + '; dropped: ' + (c.join_check.dropped.join(', ') || 'none') + '). Rerun engine rank --join-only.', 'warn'));
    }
    if (!c.candidates.length) { body.appendChild(notice('Stage 3 kept no candidate.')); return; }
    body.appendChild(table(['Rank', 'Candidate', 'Gene', 'Model', 'Phase', 'ClinVar P/LP', { text: 'Exomiser rank', num: true }, { text: 'Exomiser score', num: true }, 'Blind ranker', 'Rule hits', 'Caveats'],
      c.candidates.map(function (x) {
        var ex = x.exomiser;
        return [{ cell: num(x.priority), class: 'num' }, { cell: h('a', { class: 'id', href: '#cand-' + x.candidate_id, text: x.candidate_id }), class: 'id' },
                x.gene_symbol, label(x.model), x.phase && x.phase.status ? label(x.phase.status) : '—',
                x.clinvar_plp ? mark('yes', 'crit') : mark('no', ''),
                { cell: num(ex ? ex.rank : null), class: 'num' }, { cell: num(ex ? ex.score : null), class: 'num' },
                agreementMark(x.agreement), num(x.rule_hits.length), num(x.caveats.length)];
      }), 'candidates'));
    c.candidates.forEach(function (x) { body.appendChild(candidateArticle(x, c.rank_present)); });
  }
  function agreementMark(a) {
    return a === 'agrees' ? mark('agrees', 'good') : a === 'disagrees' ? mark('disagrees', 'warn') : a === 'not_joined' ? mark('not joined', 'warn') : mark(label(a), '');
  }
  function candidateArticle(x, rankPresent) {
    var ex = x.exomiser;
    var art = h('article', { class: 'candidate', id: 'cand-' + x.candidate_id },
      h('h3', null, num(x.priority), '. ', h('span', { class: 'id', text: x.candidate_id }), ' ', h('span', { class: 'muted', text: '· ' + dash(x.gene_symbol) + ' · ' + label(x.model) })),
      facts([
        ['Gene', h('span', null, dash(x.gene_symbol), ' ', h('span', { class: 'mono small muted', text: dash(x.gene_id) }))],
        ['Phase', x.phase ? h('span', null, label(x.phase.status), x.phase.evidence ? h('span', { class: 'muted', text: ' — ' + x.phase.evidence }) : null) : null],
        ['Rule hits', x.rule_hits.length ? h('span', null, x.rule_hits.map(function (r) { return h('span', { class: 'chip', text: r }); })) : h('span', { class: 'faint', text: 'none' })],
        ['Caveats', x.caveats.length ? h('span', null, x.caveats.map(function (r) { return h('span', { class: 'chip', text: r }); })) : h('span', { class: 'faint', text: 'none' })],
        ['Blind ranker', rankPresent ? (ex
          ? h('span', null, 'rank ', num(ex.rank), ' · score ', num(ex.score), ' · phenotype ', num(ex.phenotype_score), ' · variant ', num(ex.variant_score), ' · ', dash(ex.moi), ' · matched by ', dash(ex.match), ' ', agreementMark(x.agreement), ' ', ex.evidence ? idLink(ex.evidence) : null)
          : h('span', null, agreementMark(x.agreement), ' ', h('span', { class: 'muted', text: x.join === 'missing' ? 'not in the stage-4 join' : 'Exomiser ranked no gene for it' }))) : undefined]
      ]),
      table(['Variant', 'Consequence', 'Impact', 'HGVSc', 'HGVSp', 'GT', 'AD', { text: 'DP', num: true }, { text: 'GQ', num: true }, { text: 'AF', num: true }, 'AF source', { text: 'nhom', num: true }, 'ClinVar', { text: 'Stars', num: true }, { text: 'SpliceAI', num: true }, 'Caveats', 'Evidence'],
        x.variants.map(function (v) {
          return [{ cell: h('span', { class: 'id', text: v.key }), class: 'id' }, dash(v.consequence), dash(v.impact),
                  h('span', { class: 'mono small', text: dash(v.hgvsc) }), h('span', { class: 'mono small', text: dash(v.hgvsp) }),
                  h('span', { class: 'mono', text: dash(v.gt) }), h('span', { class: 'mono', text: dash(v.ad) }),
                  { cell: num(v.dp), class: 'num' }, { cell: num(v.gq), class: 'num' }, { cell: af(v.af_used), class: 'num' }, dash(v.af_source),
                  { cell: num(v.gnomad_nhom), class: 'num' },
                  h('span', null, v.clinvar_pathogenicity ? label(v.clinvar_pathogenicity) : '—', v.clinvar_vcv ? h('span', { class: 'mono small muted', text: ' ' + v.clinvar_vcv }) : null),
                  { cell: num(v.clinvar_stars), class: 'num' }, { cell: num(v.spliceai_ds_max), class: 'num' },
                  (v.caveats || []).length ? (v.caveats || []).map(function (r) { return h('span', { class: 'chip', text: r }); }) : h('span', { class: 'faint', text: '—' }),
                  { cell: idLinks(v.evidence, ''), class: 'ev' }];
        }), 'variants'));
    return art;
  }

  // ------------------------------------------------------------------- ranking

  function renderRanking(r, body) {
    body.appendChild(h('h3', { text: 'Blind ranking' }));
    if (!r.present) { body.appendChild(notice('Stage 4 has not run: no 04_rank/ranking.tsv or joined.json in this run.')); return; }
    body.appendChild(facts([
      ['Exomiser', h('span', null, dash(r.exomiser_version), ' · data ', dash(r.data_version))],
      ['Analysis YAML sha256', r.analysis_yaml_sha256 ? h('span', { class: 'mono small', text: r.analysis_yaml_sha256 }) : null],
      ['HPO terms given', r.hpo && r.hpo.length ? h('span', null, r.hpo.map(function (t) { return h('span', { class: 'chip', text: t }); })) : h('span', { class: 'faint', text: 'none' })],
      ['Rows', h('span', null, 'top ', num(r.top_n), ' of ', num(r.rows_total), ' ranked genes, plus every shortlist gene')],
      ['Counts', r.counts && Object.keys(r.counts).length ? h('span', { class: 'mono small', text: Object.keys(r.counts).map(function (k) { return k + ' ' + r.counts[k]; }).join(' · ') }) : null]
    ]));
    if (r.join_check && r.join_check.stale) { body.appendChild(notice('The join is stale: rerun engine rank --join-only.', 'warn')); }
    body.appendChild(table([{ text: 'Rank', num: true }, 'Gene', { text: 'Exomiser score', num: true }, { text: 'Phenotype', num: true }, { text: 'Variant', num: true }, 'MOI', { text: 'Variants', num: true }, 'Variant keys', 'Shortlist', 'Record'],
      r.rows.map(function (row) {
        return [{ cell: num(row.rank), class: 'num' }, row.gene_symbol, { cell: num(row.exomiser_score), class: 'num' }, { cell: num(row.phenotype_score), class: 'num' },
                { cell: num(row.variant_score), class: 'num' }, dash(row.moi), { cell: num(row.n_variants), class: 'num' },
                (row.variants || []).map(function (k) { return h('span', { class: 'chip', text: k }); }),
                row.in_shortlist ? h('span', null, h('span', { class: 'id', text: row.candidate_id }), row.joined === false ? [' ', mark('not joined', 'warn')] : null) : h('span', { class: 'faint', text: '—' }),
                { cell: idLink(row.evidence), class: 'ev' }];
      }), 'ranking'));
    if (r.exomiser_only && r.exomiser_only.length) {
      body.appendChild(h('h4', { text: 'Ranked by Exomiser, not on the shortlist' }));
      body.appendChild(table([{ text: 'Rank', num: true }, 'Gene', { text: 'Score', num: true }, 'MOI', 'Record'],
        r.exomiser_only.map(function (g) { return [{ cell: num(g.rank), class: 'num' }, g.gene_symbol, { cell: num(g.exomiser_score), class: 'num' }, dash(g.moi), { cell: idLink(g.evidence), class: 'ev' }]; })));
    }
  }

  // --------------------------------------------------------------------- chain

  function agentFacts(v) {
    return facts([
      ['Mode', v.manifest ? (v.dry_run ? mark('dry run — bundles and prompts only, no model was called', 'warn') : mark('live', 'live')) : h('span', { class: 'faint', text: 'no manifest' })],
      ['Model', v.manifest && !v.dry_run ? h('span', null, h('span', { class: 'mono', text: dash(v.model) }), ' · effort ', dash(v.effort)) : (v.model ? h('span', { class: 'muted' }, h('span', { class: 'mono', text: v.model }), ' (not called)') : null)],
      ['Disclosure', v.disclosure || null]
    ]);
  }
  function failuresNotice(failures, body) {
    (failures || []).forEach(function (f) {
      body.appendChild(notice('Failed on ' + f.candidate_id + ' after ' + f.turns + ' turn' + (f.turns === 1 ? '' : 's') + ': ' + f.error + (f.request_id ? ' (request ' + f.request_id + ')' : '') + ' — ' + f.file, 'crit'));
    });
  }
  function validationBlock(v, what) {
    if (!v) { return null; }
    var box = h('div', null, h('h4', { text: 'Validator' }),
      h('p', { class: 'small muted', text: v.counts ? Object.keys(v.counts).map(function (k) { return k + ' ' + v.counts[k]; }).join(' · ') : '' }));
    if (v.rejections.length) {
      box.appendChild(h('h5', { text: 'Rejected (removed from the ' + what + ')' }));
      box.appendChild(table(['Path', 'Reason'], v.rejections.map(function (r) { return [h('span', { class: 'mono small', text: r.path }), { cell: r.reason, class: 'prose' }]; })));
    } else { box.appendChild(h('p', { class: 'small', text: 'Nothing was rejected.' })); }
    if (v.disputes.length) {
      box.appendChild(h('h5', { text: 'Disputed (kept, flagged)' }));
      box.appendChild(table(['Path', 'Reason'], v.disputes.map(function (r) { return [h('span', { class: 'mono small', text: r.path }), { cell: r.reason, class: 'prose' }]; })));
    }
    if (v.notes.length) { box.appendChild(h('h5', { text: 'Notes' })); box.appendChild(list(v.notes)); }
    return box;
  }
  function references(entries) {
    if (!entries || !entries.length) { return null; }
    return h('div', null, h('h4', { text: 'References' }), h('ul', { class: 'refs' }, entries.map(function (e) {
      return h('li', null, idLink(e), ' ', h('span', { class: 'muted small', text: (e.source || '') + (e.stage ? ' · ' + e.stage : '') + (e.url ? ' · ' + e.url : ' · unresolved') }));
    })));
  }

  function renderChain(ch, body) {
    body.appendChild(h('h3', { text: 'Evidence chains' }));
    if (!ch.present) { body.appendChild(notice('Stage 5 has not run: no 05_reason/ in this run.')); return; }
    body.appendChild(agentFacts(ch));
    body.appendChild(facts([
      ['Prompt version', ch.prompt_version || null],
      ['Candidates selected', ch.candidates_selected.length ? ch.candidates_selected.map(function (c) { return h('span', { class: 'chip', text: c }); }) : null],
      ['Chains written by stage 5', num(ch.chains_written)]
    ]));
    failuresNotice(ch.failures, body);
    if (ch.notes.length) { body.appendChild(h('details', null, h('summary', { text: 'Manifest notes (' + ch.notes.length + ')' }), list(ch.notes))); }
    if (!ch.chains.length) { body.appendChild(notice(ch.dry_run ? 'A dry run writes bundles and prompts, no chain. Switch the job panel to Live mode to call the model.' : 'No chain in 05_reason/chains/.')); return; }
    ch.chains.forEach(function (chain) { body.appendChild(chainArticle(chain)); });
  }
  function chainArticle(c) {
    var refs = refMap(c.references);
    var art = h('article', { class: 'chain' },
      h('h3', null, h('span', { class: 'id', text: c.candidate_id }), ' ', h('span', { class: 'muted', text: '· ' + dash(c.gene_symbol) + ' · ' + label(c.model) })),
      c.claimed_by_manifest ? null : notice(c.manifest_note || 'not claimed by the stage-5 manifest', 'warn'),
      h('p', { class: 'small muted', text: c.file }));
    c.variants.forEach(function (v) {
      art.appendChild(h('h4', null, h('span', { class: 'id', text: v.key }), ' ', classificationMark(v.classification)));
      art.appendChild(h('p', { class: 'wide verdict' }, cited(v.summary, refs)));
      art.appendChild(table(['Code', 'Strength', 'Met', 'Justification', 'Evidence'], v.criteria.map(function (cr) {
        return [h('span', { class: 'mono', text: cr.code }), label(cr.strength), cr.met ? mark('met', 'good') : mark('not met', ''),
                { cell: cited(cr.justification, refs), class: 'prose' }, { cell: idLinks(cr.evidence, ''), class: 'ev' }];
      }), 'criteria'));
    });
    art.appendChild(h('h4', { text: 'Phase' })); art.appendChild(h('p', { class: 'wide' }, cited(c.phase_statement, refs)));
    art.appendChild(h('h4', { text: 'Mechanism hypothesis' })); art.appendChild(h('p', { class: 'wide' }, cited(c.mechanism_hypothesis, refs)));
    art.appendChild(h('h4', { text: 'Limits' })); art.appendChild(list(c.limits, null, refs));
    art.appendChild(h('h4', { text: 'What would change the call' })); art.appendChild(list(c.what_would_change_the_call, null, refs));
    art.appendChild(h('h4', { text: 'Literature' })); art.appendChild(h('p', { class: 'wide' }, idLinks(c.literature)));
    append(art, validationBlock(c.validation, 'chain'));
    append(art, references(c.references));
    return art;
  }

  // ------------------------------------------------------------------ medicine

  function renderMedicine(m, body) {
    body.appendChild(h('h3', { text: 'Medicine report' }));
    if (!m.present) { body.appendChild(notice('Stage 6 has not run: no 06_medicine/ in this run.')); return; }
    body.appendChild(agentFacts(m));
    body.appendChild(facts([
      ['Candidate', m.candidate_id ? h('span', null, h('span', { class: 'id', text: m.candidate_id }), ' · ', dash(m.gene_symbol)) : null],
      ['Stage-5 verdicts', m.stage5_verdicts.length ? m.stage5_verdicts.map(function (v) { return h('span', { class: 'nowrap' }, h('span', { class: 'id', text: v.key }), ' ', classificationMark(v.classification), ' '); }) : h('span', { class: 'faint', text: 'none in the bundle' })]
    ]));
    failuresNotice(m.failures, body);
    if (m.notes.length) { body.appendChild(h('details', null, h('summary', { text: 'Manifest notes (' + m.notes.length + ')' }), list(m.notes))); }
    if (!m.report) {
      body.appendChild(notice(m.dry_run ? 'A dry run writes the bundle and the exact prompt, no report. Switch the job panel to Live mode to call the model.' : 'No report.json in 06_medicine/.'));
      append(body, validationBlock(m.validation, 'report'));
      return;
    }
    var r = m.report, refs = refMap(r.references);
    body.appendChild(h('h4', { text: 'Mechanism' }));
    body.appendChild(claims(r.mechanism, refs));
    body.appendChild(h('h4', { text: 'Pathway targets' }));
    body.appendChild(claims(r.pathway_targets, refs));
    body.appendChild(h('h4', { text: 'Drug candidates' }));
    if (!r.candidates.length) { body.appendChild(h('p', { class: 'muted', text: 'none proposed' })); }
    r.candidates.forEach(function (d) {
      body.appendChild(h('article', { class: 'drug' },
        h('h3', null, num(d.n), '. ', d.name, ' ', d.chembl ? idLink(d.chembl) : (d.chembl_id ? h('span', { class: 'mono small muted', text: d.chembl_id }) : null)),
        facts([
          ['Mechanism of action', cited(d.mechanism_of_action, refs)],
          ['Approval status', cited(d.approval_status, refs)],
          ['Evidence', idLinks(d.evidence)],
          ['Trials', idLinks(d.trials)]
        ]),
        h('h5', { text: 'Rationale' }), h('p', { class: 'wide' }, cited(d.rationale, refs)),
        h('h5', { text: 'Counter-arguments' }), list(d.counter_arguments, null, refs)));
    });
    body.appendChild(h('h4', { text: 'Follow-up experiments' })); body.appendChild(list(r.follow_up_experiments, null, refs));
    body.appendChild(h('h4', { text: 'Limits' })); body.appendChild(list(r.limits, null, refs));
    body.appendChild(h('h4', { text: 'Literature' })); body.appendChild(h('p', { class: 'wide' }, idLinks(r.literature)));
    append(body, validationBlock(m.validation, 'report'));
    append(body, references(r.references));
  }
  function claims(items, refs) {
    if (!items || !items.length) { return h('p', { class: 'muted', text: 'none' }); }
    return h('ul', null, items.map(function (c) { return h('li', { class: 'wide' }, cited(c.statement, refs), ' ', h('span', { class: 'cites small' }, idLinks(c.evidence))); }));
  }

  // ---------------------------------------------------------------- provenance

  function renderProvenance(p, body) {
    body.appendChild(h('h3', { text: 'Provenance' }));
    p.stages.forEach(function (st) {
      var art = h('article', { class: 'manifest' }, h('h3', null, h('span', { class: 'mono muted', text: st.dir.slice(0, 2) }), ' ', st.stage));
      if (!st.manifest) { art.appendChild(h('p', { class: 'muted', text: st.present ? 'directory present, no manifest — the stage did not finish' : 'not run' })); body.appendChild(art); return; }
      var m = st.manifest;
      art.appendChild(facts([
        ['Engine', h('span', null, dash(m.engine_version), ' · ', dash(m.platform))],
        ['Started · finished', h('span', { class: 'num', text: when(m.started_at) + ' · ' + when(m.finished_at) })]
      ]));
      if (m.inputs.length) { art.appendChild(h('h5', { text: 'Inputs' })); art.appendChild(filesTable(m.inputs)); }
      if (Object.keys(m.tools).length) { art.appendChild(h('h5', { text: 'Tools' })); art.appendChild(kv(m.tools)); }
      if (Object.keys(m.counts).length) { art.appendChild(h('h5', { text: 'Counts' })); art.appendChild(kv(m.counts)); }
      art.appendChild(h('details', null, h('summary', { text: 'Parameters (' + Object.keys(m.params).length + ')' }), pre(m.params)));
      if (m.outputs.length) { art.appendChild(h('h5', { text: 'Outputs' })); art.appendChild(filesTable(m.outputs)); }
      if (m.notes.length) { art.appendChild(h('h5', { text: 'Notes' })); art.appendChild(list(m.notes)); }
      body.appendChild(art);
    });
  }
  function filesTable(entries) {
    return table(['Name', 'Path', { text: 'Bytes', num: true }, 'sha256'], entries.map(function (e) {
      return [e.name, { cell: h('span', { class: 'mono small', text: dash(e.path) }), class: 'path' }, { cell: num(e.bytes), class: 'num' },
              { cell: h('span', { class: 'mono small', text: dash(e.sha256) }), class: 'path' }];
    }), 'files');
  }


  // ---------------------------------------------------------------------- jobs

  function loadStages() {
    return api('/api/stages').then(function (doc) { state.stages = doc; })
      .catch(function (e) { state.stages = { stages: [], error: e.message }; });
  }
  function loadProviders() {
    return api('/api/providers').then(function (doc) { state.providers = doc; })
      .catch(function (e) { state.providers = { providers: {}, error: e.message }; });
  }
  function stageSpec(name) {
    var found = null;
    ((state.stages && state.stages.stages) || []).forEach(function (s) { if (s.stage === name) { found = s; } });
    return found;
  }

  function openJob(stage) {
    state.jobStage = stage;
    $('job').hidden = false;
    $('job-form').hidden = false;
    $('job-strip').hidden = true;
    var sel = clear($('job-stage'));
    STAGES.forEach(function (s) { sel.appendChild(h('option', { value: s[1], selected: s[1] === stage, text: s[0].slice(0, 2) + ' · ' + s[1] })); });
    sel.value = stage;
    renderJobForm();
    $('job').scrollIntoView({ block: 'nearest' });
  }

  function renderJobForm() {
    var stage = state.jobStage, spec = stageSpec(stage), box = clear($('job-args'));
    $('job-title').textContent = 'Job · ' + stage;
    $('job-form-msg').textContent = '';
    if (!spec) { box.appendChild(h('p', { class: 'small crit', text: 'No argument table for ' + stage + (state.stages && state.stages.error ? ': ' + state.stages.error : '') })); return; }
    var live = false, cfg = (state.stages && state.stages.config) || {};
    spec.args.forEach(function (a) {
      if (a.name === 'dry_run') { return; }  // the Live mode switch below is its inverse
      var input;
      if (a.kind === 'bool') {
        input = h('input', { type: 'checkbox', name: a.name, checked: !!a.default });
        box.appendChild(h('label', { class: 'check' }, input, a.option, ' ', h('span', { class: 'muted small', text: a.help })));
        return;
      }
      if (a.kind === 'choice') {
        input = h('select', { name: a.name, class: 'mono' }, [h('option', { value: '', text: '(default)' })].concat(a.choices.map(function (c) { return h('option', { value: c, text: c }); })));
        if (a.default) { input.value = a.default; }
      } else if (a.kind === 'int') {
        input = h('input', { type: 'number', name: a.name, class: 'mono', min: a.min, max: a.max, step: '1', value: a.default });
      } else {
        var placeholder = a.kind === 'path' ? '/path' : '';
        if (a.name === 'funnel' && cfg.funnel_bed) { placeholder = cfg.funnel_bed + ' (discovered)'; }
        input = h('input', { type: 'text', name: a.name, class: 'mono', value: a.default, placeholder: placeholder });
      }
      var field = h('label', { class: 'field', 'data-arg': a.name }, h('span', null, a.option + ' — ' + a.help), input);
      if (a.name === 'provider' || a.name === 'model' || a.name === 'effort') { field.hidden = !live; field.className += ' live-only'; }
      box.appendChild(field);
    });
    if (!spec.args.length) { box.appendChild(h('p', { class: 'small muted', text: 'This stage takes no argument from the page: it runs on the run directory alone.' })); }
    var fixed = [];
    if (stage === 'retrieve') { fixed.push(cfg.clinvar_vcf ? '--clinvar-vcf ' + cfg.clinvar_vcf : 'no ClinVar VCF discovered (see Setup)'); if (cfg.funnel_bed) { fixed.push('--funnel ' + cfg.funnel_bed); } }
    if (stage === 'rank') { fixed.push(cfg.exomiser_data ? '--exomiser-data ' + cfg.exomiser_data : 'no verified Exomiser data (see Setup)'); if (cfg.funnel_bed) { fixed.push('--regions ' + cfg.funnel_bed); } }
    if ((stage === 'retrieve' || spec.agent) && cfg.cache) { fixed.push('--cache ' + cfg.cache); }
    if (fixed.length) { box.appendChild(h('p', { class: 'small muted wide', text: 'Carried from the setup: ' + fixed.join(' · ') })); }
    var mode = $('job-mode');
    if (spec.agent) {
      mode.hidden = false;
      clear(mode);
      var sw = h('input', { type: 'checkbox', id: 'job-live' });
      mode.appendChild(h('label', { class: 'check' }, sw, h('span', null, h('b', { text: 'Live mode' }), ' — call the model (dry_run off)')));
      mode.appendChild(h('p', { id: 'job-mode-state', class: 'wide small' }));
      sw.addEventListener('change', function () { applyLive(sw.checked); });
      applyLive(false);
    } else {
      mode.hidden = true;
    }
  }

  function providerDefaults() {
    var p = state.providers || {}, st = p.providers || {};
    return { status: st, defaultProvider: p.default_provider || null, efforts: p.efforts || [], defaultEffort: p.default_effort || 'high', dotenv: p.dotenv };
  }
  function applyLive(live) {
    var mode = $('job-mode'), stateEl = $('job-mode-state'), form = $('job-form');
    mode.className = 'job-mode' + (live ? ' live' : '');
    Array.prototype.forEach.call(form.querySelectorAll('.live-only'), function (f) { f.hidden = !live; });
    var pd = providerDefaults();
    var provSel = form.elements.provider, modelIn = form.elements.model, effSel = form.elements.effort;
    if (live) {
      Array.prototype.forEach.call(provSel.options, function (o) {
        if (!o.value) { return; }
        var s = pd.status[o.value];
        o.textContent = o.value + (o.value === 'fake' ? ' (scripted, no key needed)' : s ? (s.key_present ? ' — key present' : ' — no key') : '');
      });
      if (!provSel.value && pd.defaultProvider) { provSel.value = pd.defaultProvider; }
      if (!effSel.value) { effSel.value = pd.defaultEffort; }
      var sync = function () {
        var prov = provSel.value || pd.defaultProvider || '(default)';
        var s = pd.status[prov] || {};
        var model = modelIn.value || s.default_model || '(provider default)';
        var effort = effSel.value || pd.defaultEffort;
        clear(stateEl);
        stateEl.appendChild(h('span', { class: 'state', text: 'LIVE' }));
        stateEl.appendChild(h('span', null, ' — provider ', h('b', { text: prov }), ' · model ', h('span', { class: 'mono', text: model }), ' · effort ', h('b', { text: effort }), '. ',
          prov === 'fake' ? 'The scripted client answers an empty document; no key is used.' : s.key_present ? 'The key is present in the server environment; the model will be called and the run will be paid for.' : 'No key for this provider in the server environment — the launch will fail unless the SDK finds credentials elsewhere.'));
        if (pd.dotenv && pd.dotenv.note) { stateEl.appendChild(h('span', { class: 'muted', text: ' ' + pd.dotenv.note })); }
      };
      provSel.onchange = sync; modelIn.oninput = sync; effSel.onchange = sync;
      sync();
    } else {
      clear(stateEl);
      stateEl.appendChild(h('span', { class: 'state', text: 'DRY RUN' }));
      stateEl.appendChild(h('span', { text: ' — the bundles and the exact prompts are written; no model is called, nothing is paid for.' }));
    }
  }

  function jobArgs() {
    var form = $('job-form'), spec = stageSpec(state.jobStage), args = {};
    if (!spec) { return args; }
    var live = spec.agent && $('job-live') && $('job-live').checked;
    spec.args.forEach(function (a) {
      if (a.name === 'dry_run') { args.dry_run = !live; return; }
      var el = form.elements[a.name];
      if (!el) { return; }
      if (a.kind === 'bool') { if (el.checked) { args[a.name] = true; } return; }
      if (!live && (a.name === 'provider' || a.name === 'model' || a.name === 'effort')) { return; }
      var v = String(el.value || '').trim();
      if (v === '') { return; }
      args[a.name] = a.kind === 'int' ? Number(v) : v;
    });
    return args;
  }

  $('job-form').addEventListener('submit', function (ev) {
    ev.preventDefault();
    var msg = $('job-form-msg');
    if (state.job && !isFinished(state.job)) { msg.textContent = 'A job is already running for this run.'; return; }
    var body = { stage: state.jobStage, args: jobArgs() };
    msg.textContent = 'Launching…';
    api('/api/runs/' + encodeURIComponent(state.run) + '/jobs', { method: 'POST', body: body }).then(function (job) {
      msg.textContent = '';
      startJob(job);
    }).catch(function (e) { msg.textContent = 'Refused (' + (e.status || '') + '): ' + e.message; });
  });
  $('job-stage').addEventListener('change', function () { state.jobStage = this.value; renderJobForm(); });
  $('job-close').addEventListener('click', function () { $('job').hidden = true; });
  $('job-cancel').addEventListener('click', function () {
    if (!state.job) { return; }
    api('/api/jobs/' + state.job.id + '/cancel', { method: 'POST' }).then(function (job) { state.job = job; renderJobStatus(); })
      .catch(function (e) { $('job-form-msg').textContent = 'Cancel refused: ' + e.message; });
  });

  // ----------------------------------------------------------------- pipeline

  function runPipeline(run, opts, msgEl) {
    msgEl.textContent = 'Starting the pipeline…';
    return api('/api/runs/' + encodeURIComponent(run) + '/pipeline', { method: 'POST', body: opts }).then(function (job) {
      msgEl.textContent = '';
      if (state.run !== run) { return selectRun(run).then(function () { startJob(job); }); }
      startJob(job);
    }).catch(function (e) {
      msgEl.textContent = 'Refused (' + (e.status || '') + '): ' + e.message;
      if (e.doc && e.doc.setup) { state.setupOpen = true; renderSetup(); }
      throw e;
    });
  }
  $('pipeline-run').addEventListener('click', function () {
    if (state.job && !isFinished(state.job)) { $('pipeline-msg').textContent = 'A job is already running for this run.'; return; }
    runPipeline(state.run, readOpts('run-pipeline-opts'), $('pipeline-msg')).catch(function () { /* shown */ });
  });

  function renderJobStrip(p) {
    var ol = clear($('job-strip'));
    if (!p) { ol.hidden = true; return; }
    ol.hidden = false;
    p.stages.forEach(function (s) {
      var kind = { pending: '', running: '', done: 'good', skipped: 'warn', failed: 'crit', cancelled: 'warn' }[s.status] || '';
      var meta = [];
      if (s.seconds !== null && s.seconds !== undefined && s.status !== 'pending') { meta.push(secs(s.seconds)); }
      if (s.reason) { meta.push(s.reason); }
      ol.appendChild(h('li', { class: s.status },
        h('div', { class: 'n', text: s.dir.slice(0, 2) }),
        h('div', { class: 'name', text: s.stage }),
        h('div', { class: 'status' }, mark(s.status === 'pending' ? 'queued' : s.status, kind)),
        meta.length ? h('p', { class: 'meta', text: meta.join(' · ') }) : null));
    });
  }

  function attachRunningJob() {
    if (state.job && state.job.run === state.run && !isFinished(state.job)) { return; }
    api('/api/jobs?run=' + encodeURIComponent(state.run)).then(function (doc) {
      var running = doc.jobs.filter(function (j) { return !isFinished(j); });
      if (running.length) { if (running[0].kind === 'stage') { openJob(running[0].stage); } startJob(running[0]); }
    }).catch(function () { /* the panel stays closed */ });
  }

  function stream(job, on) {
    var es = new EventSource('/api/jobs/' + job.id + '/events');
    ['status', 'line', 'stage', 'tool_call', 'transcript', 'done'].forEach(function (type) {
      es.addEventListener(type, function (e) {
        var d = JSON.parse(e.data);
        if (type === 'done') { es.close(); }
        if (on[type]) { on[type](d); }
      });
    });
    es.onerror = function () { if (isFinished(job)) { es.close(); } else if (on.interrupted) { on.interrupted(); } };
    return es;
  }

  function startJob(job) {
    if (state.es) { state.es.close(); state.es = null; }
    state.job = job;
    var log = clear($('job-log'));
    clear($('job-tools'));
    $('job-tools-wrap').hidden = !job.agent;
    $('job').hidden = false;
    $('job-form').hidden = job.kind === 'pipeline';
    $('job-title').textContent = (job.kind === 'pipeline' ? 'Pipeline' : 'Job · ' + job.stage) + ' · ' + job.id;
    renderJobStrip(job.pipeline);
    renderJobStatus();
    if (state.summary) { renderStages(state.summary); }
    state.es = stream(job, {
      status: function (d) { if (d.status && d.status !== 'cancelling') { state.job.status = d.status; } if (d.pid) { state.job.pid = d.pid; } if (d.argv && d.stage !== 'pipeline') { logLine('$ ' + d.argv.join(' '), 'sys'); } if (d.pipeline) { state.job.pipeline = d.pipeline; renderJobStrip(d.pipeline); } renderJobStatus(); },
      line: function (d) { logLine(d.text, d.stream === 'stderr' ? 'err' : ''); },
      stage: function (d) {
        var p = state.job.pipeline;
        if (p) {
          p.stages.forEach(function (s) { if (s.stage === d.stage) { s.status = d.status === 'started' ? 'running' : d.status; s.seconds = d.seconds; s.reason = d.reason; s.exit_code = d.exit_code; } });
          p.current = d.status === 'started' ? d.stage : null;
          state.job.current_stage = p.current; state.job.current_dir = p.current ? d.dir : null;
          renderJobStrip(p);
        }
        logLine('— ' + d.dir + ' · ' + d.status + (d.reason ? ' · ' + d.reason : '') + (d.seconds !== null && d.seconds !== undefined && d.status !== 'started' ? ' · ' + secs(d.seconds) : '') + (d.argv ? '\n$ ' + d.argv.join(' ') : ''), d.status === 'failed' ? 'err' : 'sys');
        if (d.status !== 'started') { loadSummary().then(function () { if (state.tab !== 'report') { renderTab(); } }); }
        else if (state.summary) { renderStages(state.summary); }
      },
      tool_call: function (d) { toolCall(d); },
      transcript: function (d) {
        logLine('transcript ' + d.file + ': ' + d.turns + ' turn' + (d.turns === 1 ? '' : 's') + ', ' + d.tool_calls + ' tool call' + (d.tool_calls === 1 ? '' : 's') + (d.failed ? ' — FAILED: ' + d.error : '') + (d.usage && d.usage.input_tokens !== null && d.usage.input_tokens !== undefined ? ' · tokens in/out ' + d.usage.input_tokens + '/' + d.usage.output_tokens : ''), d.failed ? 'err' : 'sys');
      },
      done: function (d) {
        state.job.status = d.status; state.job.exit_code = d.exit_code; state.job.manifest_present = d.manifest_present;
        if (d.stages && state.job.pipeline) { state.job.pipeline.stages = d.stages; state.job.pipeline.current = null; renderJobStrip(state.job.pipeline); }
        logLine('exit ' + d.exit_code + ' · ' + d.status + (d.manifest_present ? ' · manifest ' + d.manifest : ' · no manifest written'), d.status === 'done' ? 'sys' : 'err');
        state.es = null;
        renderJobStatus();
        loadSummary().then(function () { renderTab(); loadRuns(); });
      },
      interrupted: function () { logLine('event stream interrupted; reconnecting…', 'sys'); }
    });
  }
  function logInto(log, text, cls) {
    var atEnd = log.scrollTop + log.clientHeight >= log.scrollHeight - 4;
    log.appendChild(h('span', { class: cls || null, text: text + '\n' }));
    if (atEnd) { log.scrollTop = log.scrollHeight; }
  }
  function logLine(text, cls) { logInto($('job-log'), text, cls); }
  function toolCall(d) {
    var ol = $('job-tools');
    $('job-tools-wrap').hidden = false;
    ol.appendChild(h('li', { class: d.is_error ? 'error' : null },
      h('div', { class: 'head' }, h('span', { class: 'mono small muted', text: (d.stage ? d.stage + ' · ' : '') + d.candidate_id + ' · turn ' + dash(d.turn) }), h('span', { class: 'name', text: d.name }), d.is_error ? mark('error', 'crit') : null,
        h('span', { class: 'small muted', text: d.result_chars.toLocaleString('en-US') + ' chars' })),
      h('div', { class: 'input', text: JSON.stringify(d.input) }),
      h('details', null, h('summary', { text: 'result' }), h('pre', { text: d.result_preview + (d.result_chars > d.result_preview.length ? '\n…' : '') }))));
    ol.scrollTop = ol.scrollHeight;
  }
  function renderJobStatus() {
    var j = state.job, el = clear($('job-status'));
    if (!j) { el.textContent = 'No job launched yet.'; $('job-cancel').disabled = true; $('job-launch').disabled = false; return; }
    var kind = j.status === 'done' ? 'good' : j.status === 'failed' ? 'crit' : j.status === 'cancelled' ? 'warn' : '';
    el.appendChild(mark(j.status, kind));
    el.appendChild(h('span', null, (j.kind === 'pipeline' ? 'pipeline' : j.stage) + ' · job ' + j.id + (j.pid ? ' · pid ' + j.pid : '') + ' · '));
    if (j.kind === 'pipeline' && j.pipeline) {
      el.appendChild(j.live ? mark('LIVE', 'live') : mark('fake provider', 'warn'));
      el.appendChild(h('span', { text: ' ' + j.pipeline.provider + ' · ' + j.pipeline.model + ' · effort ' + j.pipeline.effort + ' · top ' + j.pipeline.top + ' · heap ' + j.pipeline.heap + (j.pipeline.rerun ? ' · re-run all' : '') + ' · ' }));
    } else if (j.agent) { el.appendChild(j.live ? mark('LIVE', 'live') : mark('dry run', 'warn')); el.appendChild(h('span', { text: ' · ' })); }
    el.appendChild(h('span', null, 'started ' + when(j.started_at) + (j.exit_code !== null && j.exit_code !== undefined ? ' · exit ' + j.exit_code : '')));
    el.appendChild(h('span', null, ' · ', h('a', { href: '/api/jobs/' + j.id + '/log', target: '_blank', rel: 'noopener', text: 'log file' })));
    var running = !isFinished(j);
    $('job-cancel').disabled = !running || !!j.cancel_requested;
    $('job-launch').disabled = running;
    if (state.summary) { renderStages(state.summary); }
    renderRunList();
  }

  // ------------------------------------------------------------------ new run

  function renderCasePicker() {
    var sel = clear($('case-pick')), form = $('new-run');
    sel.appendChild(h('option', { value: '', text: state.cases.length ? 'type a path below' : 'no case.yaml in the project directory — type a path' }));
    state.cases.forEach(function (c) {
      var text = c.name + ' — ' + (c.proband_id || '?') + ' · ' + (c.vcf || 'no vcf') + ' · ' + (c.hpo_count === null || c.hpo_count === undefined ? '?' : c.hpo_count) + ' HPO' + (c.error ? ' · ERROR' : '');
      sel.appendChild(h('option', { value: c.path, text: text }));
    });
    var usable = state.cases.filter(function (c) { return !c.error; });
    if (usable.length && !form.elements.case_path.value) { sel.value = usable[0].path; }
    pickCase();
  }
  function pickCase() {
    var sel = $('case-pick'), form = $('new-run'), info = clear($('case-info'));
    var c = state.cases.filter(function (x) { return x.path === sel.value; })[0];
    $('case-path-field').hidden = !!c;
    if (!c) { form.elements.name.placeholder = 'proband-yyyymmdd-hhmm (default)'; return; }
    form.elements.case_path.value = c.path;
    form.elements.name.placeholder = c.default_run_name || 'proband-yyyymmdd-hhmm';
    info.appendChild(h('span', null, 'proband ', h('b', { class: 'mono', text: dash(c.proband_id) }), ' · VCF ', h('span', { class: 'mono', text: dash(c.vcf) }), c.vcf_present === false ? ' (missing)' : '',
      ' · ', String(c.hpo_count === null ? '?' : c.hpo_count), ' HPO term' + (c.hpo_count === 1 ? '' : 's'), c.regions ? ' · regions ' + c.regions : ''));
    if (c.error) { info.appendChild(h('span', { class: 'crit', text: ' — ' + c.error })); }
  }
  $('case-pick').addEventListener('change', pickCase);

  $('new-run').addEventListener('submit', function (ev) {
    ev.preventDefault();
    var f = ev.target, msg = $('new-run-msg'), andRun = (ev.submitter || $('new-run-go')).value === 'pipeline';
    var body = { name: f.elements.name.value.trim(), case_path: f.elements.case_path.value.trim() };
    if (f.elements.regions.value.trim()) { body.regions = f.elements.regions.value.trim(); }
    if (!body.case_path) { msg.textContent = 'Pick a case file or type its path.'; return; }
    msg.textContent = 'Creating…';
    api('/api/runs', { method: 'POST', body: body }).then(function (doc) {
      msg.textContent = 'Created ' + doc.name + (andRun ? '; starting the pipeline…' : '. Run the pipeline from the run page, or one stage from the strip.');
      f.elements.name.value = ''; f.elements.regions.value = '';
      return loadRuns().then(function () { return selectRun(doc.name); }).then(function () {
        if (andRun) { return runPipeline(doc.name, readOpts('new-run-pipeline'), msg); }
        openJob('ingest');
      });
    }).catch(function (e) { msg.textContent = 'Refused: ' + e.message; });
  });

  // ---------------------------------------------------------------------- boot

  function fromHash() {
    var out = {};
    location.hash.replace(/^#/, '').split('&').forEach(function (kv) { var p = kv.split('='); if (p[0]) { out[p[0]] = decodeURIComponent(p[1] || ''); } });
    return out;
  }
  window.addEventListener('hashchange', function () {
    var hs = fromHash();
    if (hs.run && hs.run !== state.run) { selectRun(hs.run, hs.tab && TABS.some(function (t) { return t[0] === hs.tab; }) ? hs.tab : state.tab); }
    else if (hs.tab && hs.tab !== state.tab && TABS.some(function (t) { return t[0] === hs.tab; })) { state.tab = hs.tab; renderTabs(); renderTab(); }
  });

  Promise.all([loadStages(), loadProviders(), loadRuns(), loadSetup(), loadCases()]).then(function () {
    var hs = fromHash();
    var names = state.runs.map(function (r) { return r.name; });
    if (hs.tab && TABS.some(function (t) { return t[0] === hs.tab; })) { state.tab = hs.tab; }
    if (hs.run && names.indexOf(hs.run) >= 0) { selectRun(hs.run); }
    else if (names.length) { selectRun(names[0]); }
    else { renderSetup(); }
  });
  setInterval(function () { if (!document.hidden) { loadRuns(); } }, 20000);
})();
