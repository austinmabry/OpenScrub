"""Export naming/HTTP contracts and the actual embedded Save As JavaScript.

Node controls the external picker, network, disk and clock; production progress
rendering and async control flow run unchanged. No browser framework is needed.
"""
import json
from pathlib import Path
import shutil
import subprocess

import pytest

import openscrub_web as web


@pytest.mark.parametrize("suffix", [".mp4", ".mkv", ""])
def test_save_as_uses_actual_output_suffix(tmp_path, monkeypatch, suffix):
    monkeypatch.setattr(web, "JOBS_DIR", str(tmp_path))
    job = web.new_job("", {}, "road.cam.mov")
    job["phase"] = "done"
    job["output"] = str(Path(job["dir"]) / ("result" + suffix))
    Path(job["output"]).write_bytes(b"synthetic-video")
    (Path(job["dir"]) / "report.json").write_bytes(b"{}")
    try:
        client = web.app.test_client()
        public = client.get(f"/api/jobs/{job['id']}").get_json()
        video_name = "road.cam_redacted" + (suffix or ".mp4")
        assert public.get("download_name") == video_name
        assert public.get("report_name") == "road.cam_audit.json"
        for route, name, payload in [("download", video_name, b"synthetic-video"),
                                     ("report", "road.cam_audit.json", b"{}")]:
            response = client.get(f"/api/jobs/{job['id']}/{route}")
            assert response.status_code == 200
            assert response.data == payload
            assert response.headers["Content-Length"] == str(len(payload))
            assert "Content-Encoding" not in response.headers
            assert name in response.headers["Content-Disposition"]
            response.close()
    finally:
        web.JOBS.pop(job["id"], None)


HARNESS = r"""
const assert = require('node:assert/strict');
let CUR='alpha', now=0, intervals=new Map(), seq=0;
const performance={now:()=>now};
function setInterval(fn){const id=++seq;intervals.set(id,fn);return id;}
function clearInterval(id){intervals.delete(id);}
const nodes=new Map(),changes=new Map();
function node(id){
 if(!nodes.has(id))nodes.set(id,{dataset:{job:'alpha'},disabled:false,
  hidden:false,attrs:{},max:1,
  get value(){return Number(this.attrs.value||0);},set value(v){this.attrs.value=String(v);},
  get textContent(){return this.text||'';},set textContent(v){this.text=v;changes.set(id,(changes.get(id)||0)+1);},
  setAttribute(k,v){this.attrs[k]=String(v);},removeAttribute(k){delete this.attrs[k];}});
 return nodes.get(id);
}
const document={getElementById:node};
let pickerCalls=[], requests=[], writes=[], cancelled=0, aborted=0, closed=0, reads=0;
let pickerError=null, writeError=null, closeError=null, fetchError=null;
let pickerGate=null, writeGate=null, closeGate=null, readGate=null;
let length='4', encoding=null, ok=true, hasBody=true;
let chunks=[new Uint8Array(4)];
const reader={async read(){reads++;if(readGate)await readGate.promise;
 return chunks.length?{value:chunks.shift(),done:false}:{done:true};},
 async cancel(){cancelled++;},releaseLock(){}};
const writer={async write(value){writes.push(value.byteLength);if(writeGate)await writeGate.promise;
 if(writeError)throw writeError;},async close(){if(closeGate)await closeGate.promise;
 if(closeError)throw closeError;closed++;},async abort(){aborted++;}};
const window={isSecureContext:true,async showSaveFilePicker(options){pickerCalls.push(options);
 if(pickerGate)await pickerGate.promise;if(pickerError)throw pickerError;
 return {async createWritable(){return writer;}};}};
async function fetch(url){requests.push(url);if(fetchError)throw fetchError;
 return {ok,status:ok?200:423,headers:{get(k){return k.toLowerCase()==='content-length'?length:
 k.toLowerCase()==='content-encoding'?encoding:null;}},body:hasBody?{getReader(){return reader;}}:null};}
function deferred(){let resolve,reject;const promise=new Promise((a,b)=>{resolve=a;reject=b;});return {promise,resolve,reject};}
async function flush(){for(let i=0;i<25;i++)await Promise.resolve();}
async function tick(ms){now+=ms;for(const fn of [...intervals.values()])fn();await flush();}
function status(){return node('export-video-status').textContent;}
function progress(){return node('export-video-progress');}
"""


def run_js(scenario):
    node_bin = shutil.which("node")
    assert node_bin, "Node.js is required for embedded UI behavior tests (see CONTRIBUTING.md)"
    # This is the real evaluated Python string, not an independently copied JS helper.
    source = web.APP_JS.split("function shellHtml(j)", 1)[1].split("async function refresh()", 1)[0]
    source = "function shellHtml(j)" + source
    script = HARNESS + "\n" + source + r"""
assert.equal(typeof saveExportAs,'function','Save As must be implemented');
EXPORT_NAMES={id:'alpha',video:'road.cam_redacted.mkv',report:'road.cam_audit.json'};
let completed=false;
process.on('beforeExit',()=>assert.equal(completed,true,'async scenario did not finish'));
(async()=>{
""" + scenario + "\n})().then(()=>{completed=true;}).catch(e=>{console.error(e);process.exitCode=1;});"
    result = subprocess.run([node_bin, "-e", script], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr


def test_picker_cancel_does_not_fetch():
    run_js(r"""
pickerError=Object.assign(new Error('cancel'),{name:'AbortError'});
await saveExportAs('video',node('save-video'));
assert.equal(pickerCalls[0].suggestedName,'road.cam_redacted.mkv');
assert.deepEqual(requests,[]);assert.equal(node('save-video').disabled,false);
assert.equal(intervals.size,0);assert.doesNotMatch(status(),/failed|Saved/i);
""")


def test_job_switch_keeps_original_export_target():
    run_js(r"""
pickerGate=deferred();const pending=saveExportAs('video',node('save-video'));
assert.equal(pickerCalls.length,1); // picker invoked synchronously in the gesture
CUR='beta';EXPORT_NAMES={id:'beta',video:'other_redacted.mp4',report:'other_audit.json'};
node('export-video').dataset.job='beta';pickerGate.resolve();await pending;
assert.deepEqual(requests,['api/jobs/alpha/download']);
assert.equal(pickerCalls[0].suggestedName,'road.cam_redacted.mkv');
""")


def test_save_progress_counts_completed_writes():
    run_js(r"""
chunks=[new Uint8Array(4),new Uint8Array(6)];length='10';writeGate=deferred();closeGate=deferred();
const pending=saveExportAs('video',node('save-video'));await flush();await tick(250);
assert.equal(progress().value,0);assert.equal(reads,1); // backpressure
writeGate.resolve();await flush();await tick(250);
assert.equal(progress().value,10);assert.equal(progress().max,10);
assert.match(status(),/10 B \/ 10 B/);assert.doesNotMatch(status(),/Saved/);
closeGate.resolve();await pending;assert.match(status(),/Saved/);assert.equal(closed,1);
assert.equal(intervals.size,0);
""")


@pytest.mark.parametrize("header,encoding", [(None, None), ("bad", None), ("-1", None),
                                              ("9007199254740992", None), ("4", "gzip")])
def test_save_progress_unknown_total(header, encoding):
    run_js("length=" + json.dumps(header) + ";encoding=" + json.dumps(encoding) + r""";
closeGate=deferred();const pending=saveExportAs('video',node('save-video'));await flush();await tick(1000);
assert.match(status(),/4 B/);assert.match(status(),/elapsed/);
assert.doesNotMatch(status(),/%|left|\//);assert.equal(progress().attrs.value,undefined);
closeGate.resolve();await pending;assert.match(status(),/Saved/);
""")


def test_save_progress_waiting_and_finalizing():
    run_js(r"""
chunks=[new Uint8Array(4),new Uint8Array(4)];length='8';writeGate=deferred();closeGate=deferred();
const pending=saveExportAs('video',node('save-video'));await flush();await tick(3000);
writeGate.resolve();await flush();await tick(250);
assert.match(status(),/Finalizing/);assert.doesNotMatch(status(),/Saved|left/);
assert.match(status(),/elapsed/);closeGate.resolve();await pending;
assert.match(status(),/Saved/);assert.equal(intervals.size,0);
""")


def test_save_progress_waiting_hides_stale_eta():
    run_js(r"""
chunks=[new Uint8Array(4),new Uint8Array(4),new Uint8Array(4)];length='12';writeGate=deferred();
const pending=saveExportAs('video',node('save-video'));await flush();await tick(3000);
const first=writeGate;writeGate=deferred();first.resolve();await flush();await tick(250);
assert.equal(progress().value,4);assert.match(status(),/left/);
await tick(5000);assert.match(status(),/Waiting for data or disk/);
assert.match(status(),/elapsed/);assert.doesNotMatch(status(),/left/);assert.equal(progress().value,4);
writeGate.resolve();await pending;
""")


def test_save_progress_stays_with_original_job():
    run_js(r"""
writeGate=deferred();const pending=saveExportAs('video',node('save-video'));await flush();
CUR='beta';node('export-video').dataset.job='beta';node('export-video-status').textContent='beta status';
await tick(5500);assert.equal(status(),'beta status');
CUR='alpha';node('export-video').dataset.job='alpha';renderExportProgress();
assert.match(status(),/Waiting/);writeGate.resolve();await pending;
assert.match(status(),/Saved/);
// Ordinary refresh/remount restores the completed state, and a repeat is allowed.
node('export-video-status').textContent='';renderExportProgress();assert.match(status(),/Saved/);
chunks=[new Uint8Array(4)];await saveExportAs('video',node('save-video'));
assert.equal(closed,2);assert.equal(intervals.size,0);
""")


@pytest.mark.parametrize("scenario", [
    "length='0';chunks=[];await saveExportAs('video',node('save-video'));assert.match(status(),/Saved/);assert.equal(closed,1);",
    "length='5';await saveExportAs('video',node('save-video'));assert.match(status(),/failed/i);assert.equal(closed,0);assert.equal(aborted,1);",
    "writeError=new Error('disk full');await saveExportAs('video',node('save-video'));assert.match(status(),/disk full/);assert.equal(cancelled,1);assert.equal(aborted,1);assert.equal(closed,0);",
    "closeError=new Error('close denied');await saveExportAs('video',node('save-video'));assert.match(status(),/close denied/);assert.equal(aborted,1);assert.doesNotMatch(status(),/Saved/);",
    "fetchError=new Error('network lost');await saveExportAs('video',node('save-video'));assert.match(status(),/network lost/);",
    "ok=false;await saveExportAs('video',node('save-video'));assert.match(status(),/locked/);",
    "hasBody=false;await saveExportAs('video',node('save-video'));assert.match(status(),/stream/);",
])
def test_save_progress_completion_and_failures(scenario):
    run_js(scenario + "assert.equal(intervals.size,0);assert.equal(node('save-video').disabled,false);")


def test_save_progress_throttles_visuals_and_live_announcements():
    run_js(r"""
readGate=deferred();const pending=saveExportAs('video',node('save-video'));await flush();
await tick(250);const visual=changes.get('export-video-status'),live=changes.get('export-video-live');
await tick(50);assert.equal(changes.get('export-video-status'),visual);
await tick(200);assert.ok(changes.get('export-video-status')>visual);
assert.equal(changes.get('export-video-live'),live);
await tick(1750);assert.ok(changes.get('export-video-live')>live);
readGate.resolve();await pending;
""")


def test_save_report_and_direct_download_feedback():
    run_js(r"""
await saveExportAs('report',node('save-report'));
assert.deepEqual(requests,['api/jobs/alpha/report']);
assert.equal(pickerCalls[0].suggestedName,'road.cam_audit.json');
assert.match(node('export-report-status').textContent,/Saved/);
exportDownloadStarted('video');assert.match(status(),/Download started — see browser downloads/);
assert.equal(progress().hidden,true);assert.doesNotMatch(status(),/Saved|%|left/);
""")


def test_save_progress_rate_uses_recent_ten_seconds():
    run_js(r"""
length='200000';chunks=[new Uint8Array(4000),new Uint8Array(100000),new Uint8Array(96000)];
writeGate=deferred();const pending=saveExportAs('video',node('save-video'));await flush();
await tick(3000);let gate=writeGate;writeGate=deferred();gate.resolve();await flush();
await tick(13000);gate=writeGate;writeGate=deferred();gate.resolve();await flush();await tick(250);
assert.match(status(),/10.0 KB\/s/);assert.equal(progress().value,104000);
writeGate.resolve();await pending;
""")


@pytest.mark.parametrize("picker", [True, False])
def test_download_stays_a_button_without_save_picker(picker):
    """Without the picker (Safari, Firefox, iPhone) Download is the only way
    out, so it must stay a full button rather than a small text link."""
    node_bin = shutil.which("node")
    assert node_bin, "Node.js is required for embedded UI behavior tests (see CONTRIBUTING.md)"
    source = "let EXPORT_NAMES" + web.APP_JS.split("let EXPORT_NAMES", 1)[1].split("function exportDownloadStarted", 1)[0]
    script = ("const window={isSecureContext:true" + (",showSaveFilePicker(){}" if picker else "")
              + "};\n" + source + "\nprocess.stdout.write(JSON.stringify("
              "[exportControls('alpha','video'),exportControls('alpha','report')]));")
    result = subprocess.run([node_bin, "-e", script], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    video, report = json.loads(result.stdout)
    for html, label in [(video, "redacted video"), (report, "audit report")]:
        assert 'href="api/jobs/alpha/' in html
        has_button = f">Download {label}</button>" in html
        assert has_button is (not picker)
        assert (f">Save {label} as…</button>" in html) is picker
