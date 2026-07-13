"""Zone Management page for openscrub_web. Kept as a raw string in its own
module so JavaScript escape sequences can never be mangled by patch layers."""

ZONES_PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>OpenScrub — Detection Zones</title>
<link rel="icon" href="favicon.ico"><style>
:root{--bg:#f5f6f8;--card:#fff;--ink:#1f2937;--mut:#6b7280;--acc:#2563eb}
*{box-sizing:border-box}
body{font:15px/1.45 system-ui,Segoe UI,Roboto,sans-serif;margin:0;
 background:var(--bg);color:var(--ink)}
header{background:#111827;color:#fff;padding:8px 16px;display:flex;
 align-items:center;gap:14px}
header b{font-weight:600}
header img{height:34px;display:block}
header a{color:#93c5fd;text-decoration:none;font-size:14px}
main{max-width:1400px;margin:0 auto;padding:12px;display:grid;
 grid-template-columns:290px 1fr;gap:12px}
@media(max-width:900px){main{grid-template-columns:1fr}}
.card{background:var(--card);border-radius:12px;padding:14px;
 box-shadow:0 1px 3px rgba(0,0,0,.08)}
h2{margin:0 0 10px;font-size:16px}
.warnbox{grid-column:1/-1;background:linear-gradient(90deg,#fffbeb,#fef3c7);
 border:1px solid #f59e0b;border-radius:12px;padding:12px 16px;font-size:13.5px}
.warnbox b{color:#92400e}
.chips{display:flex;flex-direction:column;gap:6px}
.chip{display:flex;align-items:center;gap:10px;padding:8px 10px;
 border-radius:9px;border:2px solid transparent;cursor:pointer;
 transition:all .15s ease;background:#f9fafb}
.chip:hover{background:#f3f4f6;transform:translateX(2px)}
.chip.active{border-color:var(--c);background:#fff;
 box-shadow:0 0 0 3px color-mix(in srgb,var(--c) 18%,transparent)}
.chip .dot{width:14px;height:14px;border-radius:4px;background:var(--c);
 box-shadow:0 0 6px color-mix(in srgb,var(--c) 55%,transparent)}
.chip .n{margin-left:auto;font-size:12px;color:var(--mut);
 background:#eef2f7;border-radius:10px;padding:1px 8px}
.chip.viewall{border-style:dashed;border-color:#d1d5db;justify-content:center;
 color:var(--mut)}
.chip.viewall.active{border-color:#9ca3af;color:var(--ink)}
.tools{display:flex;gap:6px;margin:10px 0}
.tools button{flex:1}
button{background:var(--acc);color:#fff;border:0;border-radius:8px;
 padding:8px 12px;font:inherit;cursor:pointer;transition:filter .12s}
button:hover{filter:brightness(1.08)}
button.sec{background:#6b7280}button.danger{background:#dc2626}
button.tog{background:#e5e7eb;color:var(--ink)}
button.tog.on{background:var(--acc);color:#fff}
select,input[type=range]{width:100%}
select{padding:7px;border:1px solid #d1d5db;border-radius:7px;font:inherit}
label{display:block;margin:8px 0 3px;font-size:12.5px;color:var(--mut)}
.stage{background:linear-gradient(160deg,#111827,#1e293b);border-radius:12px;
 padding:14px;display:flex;flex-direction:column;gap:8px}
.stagebar{display:flex;align-items:center;gap:10px;color:#cbd5e1;font-size:12.5px}
#saved{margin-left:auto;font-size:12px;padding:2px 10px;border-radius:10px;
 background:#064e3b;color:#6ee7b7;transition:opacity .3s}
#cwrap{position:relative;align-self:center;max-width:100%;
 box-shadow:0 8px 30px rgba(0,0,0,.45);border-radius:6px;overflow:hidden}
#bg,#cv{display:block;max-width:100%}
#cv{position:absolute;left:0;top:0;touch-action:none}
#coords{font:12px ui-monospace,Consolas,monospace;color:#93c5fd;min-width:210px}
#tl{display:flex;align-items:center;gap:10px;padding:2px 4px;user-select:none}
.tltime{font:12px ui-monospace,Consolas,monospace;color:#cbd5e1;min-width:44px;text-align:center}
#tltrack{position:relative;flex:1;height:22px;cursor:pointer}
#tltrack::before{content:"";position:absolute;left:0;right:0;top:8px;height:6px;
 border-radius:3px;background:#334155}
#tlticks{position:absolute;left:0;right:0;top:8px;height:6px}
.tick{position:absolute;top:0;width:1px;height:6px;background:rgba(255,255,255,.18)}
#tlfill{position:absolute;left:0;top:8px;height:6px;border-radius:3px;
 background:linear-gradient(90deg,#3b82f6,#60a5fa);width:0}
#tlhead{position:absolute;top:3px;width:16px;height:16px;border-radius:50%;
 background:#fff;border:3px solid #3b82f6;box-shadow:0 1px 6px rgba(0,0,0,.5);
 margin-left:-8px;left:0}
#tlbub{position:absolute;bottom:26px;transform:translateX(-50%);background:#0f172a;
 color:#e2e8f0;font:11px ui-monospace,monospace;padding:3px 7px;border-radius:6px;
 display:none;white-space:nowrap;border:1px solid #334155}
#tl.off{opacity:.35;pointer-events:none}
.hint{font-size:12px;color:#94a3b8}
kbd{background:#334155;color:#e2e8f0;border-radius:4px;padding:0 5px;
 font:11px ui-monospace,monospace}
</style></head><body>
<header><img src="logo_dark.png" alt="OpenScrub">
 <b>Detection Zones</b>
 <a href="./" style="margin-left:auto">← back to jobs</a></header>
<main>
<div class="warnbox"><b>⚠ Read before using zones.</b> Zones restrict where
each category is detected. <b>Anything outside a category's zones will NOT be
blurred — even if the software sees it.</b> A name in a popup, a DOB in an
unexpected corner, a face outside its zone: all exposed. Zones trade recall
for precision. Use them only for layouts you know cold, run a
<i>preview-mode</i> pass after changing them, and watch for the
<i>ZONE&nbsp;WARNING</i> in the job log — it counts PHI that was detected but
left unblurred because it fell outside your zones. Categories with no zones
remain full-frame (the safe default). Zones are stored in resolution-independent
coordinates and apply to <b>all</b> jobs while enabled.</div>

<div class="card">
<h2>Classes</h2>
<div class="chips" id="chips"></div>
<div class="tools">
 <button class="tog on" id="mDraw" onclick="setMode('draw')">✏ Draw</button>
 <button class="tog" id="mSel" onclick="setMode('select')">⬚ Select</button>
</div>
<div class="tools">
 <button class="sec" onclick="undo()" title="Ctrl+Z">↶ Undo</button>
 <button class="sec" onclick="clearClass()">Clear class</button>
 <button class="danger" onclick="resetAll()">Reset all</button>
</div>
<label>Background — video file from this device (recommended)</label>
<input type="file" id="localvid" accept="video/*" onchange="loadLocal()">
<label>…or a frame from an uploaded job (any phase)</label>
<select id="jobsel" onchange="pickJob()"></select>
<input type="hidden" id="tslide" min="0" max="10" value="0" disabled>
<label>…or a reference screenshot</label>
<input type="file" id="refimg" accept="image/*" onchange="loadRef()">
<p class="hint" style="margin-top:12px">
<b>Draw:</b> click to anchor a corner, move, click again to set the opposite
corner (or click-drag-release). <kbd>Esc</kbd> cancels.<br>
<b>Select:</b> click a zone of the active class — drag to move, drag a corner
handle to resize, <kbd>Del</kbd> to remove.<br>
Overlapping zones of the same class merge into one shape.</p>
</div>

<div class="stage">
<div class="stagebar">
 <span id="modeinfo">Editing: —</span>
 <span id="coords"></span>
 <span id="saved" style="opacity:0">Saved ✓</span>
</div>
<div id="cwrap"><img id="bg"><video id="bgv" muted playsinline
 style="display:none"></video><canvas id="cv"></canvas></div>
<div id="tl" class="off"><span class="tltime" id="tlcur">0:00</span>
 <div id="tltrack"><div id="tlticks"></div><div id="tlfill"></div>
  <div id="tlhead"></div><div id="tlbub">0:00</div></div>
 <span class="tltime" id="tldur">0:00</span></div>
<div class="stagebar"><span class="hint" id="reshint"></span></div>
</div>
</main>
<script>
const CATS={name:"#3b82f6",dob:"#22c55e",phone:"#f59e0b",ssn:"#ef4444",
            mrn:"#8b5cf6",email:"#14b8a6",address:"#f97316",card:"#db2777",
            apikey:"#0891b2",ipaddr:"#65a30d",plate:"#7c3aed",face:"#ec4899",ignore:"#334155"};
let zones={},active=null,mode="draw",undoStack=[],saveT=null;
let anchor=null,floatPt=null,dragKind=null,dragIdx=-1,dragOff=null,selIdx=-1,beforeDrag=null;
let natW=1280,natH=720,DUR=10,CURJOB=null,BGMODE="img";

const bg=document.getElementById("bg"),bgv=document.getElementById("bgv"),
      cv=document.getElementById("cv"),ctx=cv.getContext("2d");
function setBgMode(m){BGMODE=m;
 bg.style.display=m==="img"?"block":"none";
 bgv.style.display=m==="video"?"block":"none";}

function chips(){
 const el=document.getElementById("chips");
 el.innerHTML=Object.entries(CATS).map(([c,col])=>
  `<div class="chip ${active===c?"active":""}" style="--c:${col}"
    onclick="setActive('${c}')"><span class="dot"></span>${c==="ignore"?"ignore (never blur)":c}
    <span class="n">${(zones[c]||[]).length}</span></div>`).join("")
  +`<div class="chip viewall ${active===null?"active":""}"
    onclick="setActive(null)">👁 view all classes</div>`;
 document.getElementById("modeinfo").textContent=
   active?("Editing: "+active):"Viewing all classes (pick one to edit)";
}
function setActive(c){active=c;selIdx=-1;anchor=null;chips();draw();}
function setMode(m){mode=m;selIdx=-1;anchor=null;
 document.getElementById("mDraw").classList.toggle("on",m==="draw");
 document.getElementById("mSel").classList.toggle("on",m==="select");
 cv.style.cursor=m==="draw"?"crosshair":"default";draw();}

function pushUndo(){undoStack.push(JSON.stringify(zones));
 if(undoStack.length>40)undoStack.shift();}
function undo(){if(!undoStack.length)return;
 zones=JSON.parse(undoStack.pop());selIdx=-1;chips();draw();save();}
function clearClass(){if(!active)return alert("Pick a class first.");
 if(!confirm("Remove all "+active+" zones? The category returns to full-frame detection."))return;
 pushUndo();zones[active]=[];chips();draw();save();}
function resetAll(){
 if(!confirm("Remove ALL zones for ALL classes? Every category returns to full-frame detection."))return;
 pushUndo();zones={};chips();draw();save();}

// ---------- geometry ----------
function toNorm(px,py){return[px/cv.width,py/cv.height];}
function rectPx(r){return[r[0]*cv.width,r[1]*cv.height,
                          r[2]*cv.width,r[3]*cv.height];}
function pointer(e){const b=cv.getBoundingClientRect();
 return[(e.clientX-b.left)*cv.width/b.width,
        (e.clientY-b.top)*cv.height/b.height];}

// ---------- rendering: union fill + union outline per class ----------
function unionLayer(rects,color,extra){
 const m=document.createElement("canvas");m.width=cv.width;m.height=cv.height;
 const mc=m.getContext("2d");mc.fillStyle="#fff";
 for(const r of rects){const[x1,y1,x2,y2]=rectPx(r);
  mc.fillRect(Math.min(x1,x2),Math.min(y1,y2),Math.abs(x2-x1),Math.abs(y2-y1));}
 if(extra){const[x1,y1,x2,y2]=extra;
  mc.fillRect(Math.min(x1,x2),Math.min(y1,y2),Math.abs(x2-x1),Math.abs(y2-y1));}
 // tinted fill
 const f=document.createElement("canvas");f.width=cv.width;f.height=cv.height;
 const fc=f.getContext("2d");fc.drawImage(m,0,0);
 fc.globalCompositeOperation="source-in";fc.fillStyle=color;
 fc.fillRect(0,0,f.width,f.height);
 ctx.globalAlpha=0.16;ctx.drawImage(f,0,0);ctx.globalAlpha=1;
 // outline ring = dilate(mask) - mask, tinted
 const d=document.createElement("canvas");d.width=cv.width;d.height=cv.height;
 const dc=d.getContext("2d");
 for(const[ox,oy] of [[-2,0],[2,0],[0,-2],[0,2],[-2,-2],[2,2],[-2,2],[2,-2]])
  dc.drawImage(m,ox,oy);
 dc.globalCompositeOperation="destination-out";dc.drawImage(m,0,0);
 dc.globalCompositeOperation="source-in";dc.fillStyle=color;
 dc.fillRect(0,0,d.width,d.height);
 ctx.globalAlpha=0.95;ctx.drawImage(d,0,0);ctx.globalAlpha=1;
}
function draw(){
 ctx.clearRect(0,0,cv.width,cv.height);
 const show=active?[active]:Object.keys(CATS);
 for(const c of show){
  const rects=zones[c]||[];
  let extra=null;
  if(c===active&&anchor&&floatPt)
   extra=[anchor[0],anchor[1],floatPt[0],floatPt[1]];
  if(rects.length||extra)unionLayer(rects,CATS[c],extra);
 }
 // selection adorners
 if(active&&mode==="select"&&selIdx>=0&&zones[active]&&zones[active][selIdx]){
  const[x1,y1,x2,y2]=rectPx(zones[active][selIdx]);
  ctx.setLineDash([6,4]);ctx.strokeStyle="#fff";ctx.lineWidth=1.5;
  ctx.strokeRect(x1,y1,x2-x1,y2-y1);ctx.setLineDash([]);
  for(const[hx,hy] of [[x1,y1],[x2,y1],[x1,y2],[x2,y2]]){
   ctx.fillStyle=CATS[active];ctx.strokeStyle="#fff";ctx.lineWidth=2;
   ctx.fillRect(hx-6,hy-6,12,12);ctx.strokeRect(hx-6,hy-6,12,12);}
 }
}

// ---------- interaction ----------
function hitHandle(p,r){const[x1,y1,x2,y2]=rectPx(r);
 const hs=[[x1,y1,0],[x2,y1,1],[x1,y2,2],[x2,y2,3]];
 for(const[hx,hy,i] of hs)
  if(Math.abs(p[0]-hx)<9&&Math.abs(p[1]-hy)<9)return i;
 return -1;}
function hitRect(p,r){const[x1,y1,x2,y2]=rectPx(r);
 return p[0]>=Math.min(x1,x2)&&p[0]<=Math.max(x1,x2)
     &&p[1]>=Math.min(y1,y2)&&p[1]<=Math.max(y1,y2);}

let downAt=null,moved=false;
cv.addEventListener("pointerdown",e=>{
 if(!active){flashPickClass();return;}
 const p=pointer(e);downAt=[...p,Date.now()];moved=false;
 cv.setPointerCapture(e.pointerId);
 if(mode==="select"){
  const rs=zones[active]||[];
  if(selIdx>=0&&rs[selIdx]!==undefined){
   const h=hitHandle(p,rs[selIdx]);
   if(h>=0){pushUndo();beforeDrag=[...zones[active][selIdx]];dragKind="handle";dragIdx=h;return;}}
  for(let i=rs.length-1;i>=0;i--)
   if(hitRect(p,rs[i])){selIdx=i;pushUndo();beforeDrag=[...rs[i]];dragKind="move";
    const[x1,y1]=rectPx(rs[i]);dragOff=[p[0]-x1,p[1]-y1];draw();return;}
  selIdx=-1;dragKind=null;draw();
 }else{ // draw
  if(anchor){commitRect(p);}
  else{anchor=p;floatPt=p;}
 }
});
cv.addEventListener("pointermove",e=>{
 const p=pointer(e);
 const[nx,ny]=toNorm(p[0],p[1]);
 document.getElementById("coords").textContent=
  `x ${nx.toFixed(3)}  y ${ny.toFixed(3)}   (${Math.round(nx*natW)}, ${Math.round(ny*natH)} px)`;
 if(downAt&&(Math.abs(p[0]-downAt[0])>5||Math.abs(p[1]-downAt[1])>5))moved=true;
 if(mode==="draw"&&anchor){floatPt=p;draw();return;}
 if(mode==="select"&&dragKind&&selIdx>=0){
  const r=zones[active][selIdx],[x1,y1,x2,y2]=rectPx(r);
  if(dragKind==="move"){
   const w=x2-x1,h=y2-y1,nx1=p[0]-dragOff[0],ny1=p[1]-dragOff[1];
   zones[active][selIdx]=[...toNorm(nx1,ny1),...toNorm(nx1+w,ny1+h)];
  }else{
   const c=[[x1,y1],[x2,y1],[x1,y2],[x2,y2]];c[dragIdx]=p;
   const xs=[c[0][0],c[1][0],c[2][0],c[3][0]],ys=[c[0][1],c[1][1],c[2][1],c[3][1]];
   if(dragIdx===0){zones[active][selIdx]=[...toNorm(p[0],p[1]),...toNorm(x2,y2)];}
   if(dragIdx===1){zones[active][selIdx]=[...toNorm(x1,p[1]),...toNorm(p[0],y2)];}
   if(dragIdx===2){zones[active][selIdx]=[...toNorm(p[0],y1),...toNorm(x2,p[1])];}
   if(dragIdx===3){zones[active][selIdx]=[...toNorm(x1,y1),...toNorm(p[0],p[1])];}
  }
  normRect(zones[active][selIdx]);
 if(clipToBarriers(zones[active][selIdx],active)===null){
  zones[active][selIdx]=beforeDrag?[...beforeDrag]:zones[active][selIdx];}
 draw();save();
 }
});
cv.addEventListener("pointerup",e=>{
 const p=pointer(e);
 if(mode==="draw"&&anchor&&moved){commitRect(p);}
 if(mode==="select"){dragKind=null;}
 downAt=null;
});
function rectsHit(a,b){return a[0]<b[2]&&a[2]>b[0]&&a[1]<b[3]&&a[3]>b[1];}
function barriersFor(cls){
 let out=[];
 for(const c in zones){
  if(cls==="ignore"?c!=="ignore":c==="ignore")out=out.concat(zones[c]||[]);
 }
 return out;
}
function clipToBarriers(r,cls){
 /* ignore zones and detection zones may never overlap: the edge of one
    group is a hard barrier for the other. Shrink the rect along whichever
    edge loses the least area; null = nothing legal remains. */
 const bars=barriersFor(cls);
 for(let guard=0;guard<16;guard++){
  const hit=bars.find(b=>rectsHit(r,b));
  if(!hit)return r;
  const cands=[];
  if(hit[0]>r[0])cands.push([r[0],r[1],hit[0],r[3]]);
  if(hit[2]<r[2])cands.push([hit[2],r[1],r[2],r[3]]);
  if(hit[1]>r[1])cands.push([r[0],r[1],r[2],hit[1]]);
  if(hit[3]<r[3])cands.push([r[0],hit[3],r[2],r[3]]);
  const ok=cands.filter(c=>c[2]-c[0]>0.005&&c[3]-c[1]>0.005)
   .sort((a,b)=>(b[2]-b[0])*(b[3]-b[1])-(a[2]-a[0])*(a[3]-a[1]))[0];
  if(!ok)return null;
  r[0]=ok[0];r[1]=ok[1];r[2]=ok[2];r[3]=ok[3];
 }
 return r;
}
function commitRect(p){
 const r=[...toNorm(anchor[0],anchor[1]),...toNorm(p[0],p[1])];
 anchor=null;floatPt=null;
 normRect(r);
 if(clipToBarriers(r,active)===null){
  document.getElementById("modeinfo").textContent=
   "⚠ blocked — ignore zones and detection zones cannot overlap";
  setTimeout(chips,1800);draw();return;
 }
 if((r[2]-r[0])*cv.width<8||(r[3]-r[1])*cv.height<8){draw();return;}
 pushUndo();(zones[active]=zones[active]||[]).push(r);
 chips();draw();save();
}
function normRect(r){
 const x1=Math.min(r[0],r[2]),x2=Math.max(r[0],r[2]),
       y1=Math.min(r[1],r[3]),y2=Math.max(r[1],r[3]);
 r[0]=Math.max(0,x1);r[1]=Math.max(0,y1);
 r[2]=Math.min(1,x2);r[3]=Math.min(1,y2);
}
document.addEventListener("keydown",e=>{
 if(e.key==="Escape"){anchor=null;floatPt=null;draw();}
 if((e.key==="Delete"||e.key==="Backspace")&&mode==="select"&&active&&selIdx>=0){
  pushUndo();zones[active].splice(selIdx,1);selIdx=-1;chips();draw();save();
  e.preventDefault();}
 if(e.ctrlKey&&e.key.toLowerCase()==="z"){undo();e.preventDefault();}
});
function flashPickClass(){
 const el=document.getElementById("modeinfo");
 el.textContent="⚠ pick a class on the left to start editing";
 setTimeout(chips,1600);
}

// ---------- persistence ----------
async function load(){
 zones=await (await fetch("api/zones")).json();
 chips();draw();
}
function save(){
 clearTimeout(saveT);
 const s=document.getElementById("saved");
 s.textContent="Saving…";s.style.opacity=1;
 saveT=setTimeout(async()=>{
  await fetch("api/zones",{method:"POST",
   headers:{"Content-Type":"application/json"},body:JSON.stringify(zones)});
  s.textContent="Saved ✓";
  setTimeout(()=>{s.style.opacity=0;},1200);
 },500);
}

// ---------- background ----------
function fitCanvas(){
 const el=BGMODE==="video"?bgv:bg;
 cv.width=el.clientWidth||960;cv.height=el.clientHeight||540;
 document.getElementById("reshint").textContent=
  `background: ${natW}×${natH} — zones are stored resolution-independent (0–1) and scale to each video`;
 draw();
}
bg.onload=()=>{setBgMode("img");natW=bg.naturalWidth;natH=bg.naturalHeight;fitCanvas();};
bgv.addEventListener("loadedmetadata",()=>{
 setBgMode("video");natW=bgv.videoWidth;natH=bgv.videoHeight;
 DUR=bgv.duration||10;
 const sl=document.getElementById("tslide");
 sl.max=Math.max(0.5,DUR-0.1).toFixed(1);sl.value=0;sl.disabled=false;tlSync();
 requestAnimationFrame(fitCanvas);
});
bgv.addEventListener("loadeddata",fitCanvas);
function loadLocal(){
 const f=document.getElementById("localvid").files[0];
 if(!f)return;
 CURJOB=null;document.getElementById("jobsel").value="";
 bgv.src=URL.createObjectURL(f);
}
function loadRef(){
 const f=document.getElementById("refimg").files[0];
 if(!f)return;
 CURJOB=null;document.getElementById("jobsel").value="";
 document.getElementById("tslide").disabled=true;
 bg.src=URL.createObjectURL(f);
}
window.addEventListener("resize",fitCanvas);
function gridBg(){
 const g=document.createElement("canvas");g.width=1280;g.height=720;
 const gc=g.getContext("2d");gc.fillStyle="#0f172a";gc.fillRect(0,0,1280,720);
 gc.strokeStyle="#1e293b";
 for(let x=0;x<1280;x+=80){gc.beginPath();gc.moveTo(x,0);gc.lineTo(x,720);gc.stroke();}
 for(let y=0;y<720;y+=80){gc.beginPath();gc.moveTo(0,y);gc.lineTo(1280,y);gc.stroke();}
 bg.src=g.toDataURL();
}
async function jobs(){
 const js=await (await fetch("api/jobs")).json();
 const sel=document.getElementById("jobsel");
 sel.innerHTML=`<option value="">— grid (no video) —</option>`+
  js.map(j=>`<option value="${j.id}">${j.name} (${j.phase})</option>`).join("");
}
async function pickJob(){
 CURJOB=document.getElementById("jobsel").value||null;
 const sl=document.getElementById("tslide");
 if(!CURJOB){sl.disabled=true;tlSync();setBgMode("img");gridBg();return;}
 const d=await (await fetch(`api/jobs/${CURJOB}/mediainfo`)).json();
 DUR=d.duration;natW=d.width;natH=d.height;
 sl.max=Math.max(0.5,DUR-0.2).toFixed(1);sl.disabled=false;
 scrub();
}
const tltrack=document.getElementById("tltrack"),tlfill=document.getElementById("tlfill"),
      tlhead=document.getElementById("tlhead"),tlbub=document.getElementById("tlbub"),
      tlcur=document.getElementById("tlcur"),tldur=document.getElementById("tldur"),
      tlbox=document.getElementById("tl");
function tfmt(t){t=Math.max(0,+t||0);const m=Math.floor(t/60),s=Math.floor(t%60);
 return m+":"+String(s).padStart(2,"0");}
let tlThrottle=null,tlDrag=false;
function tlSync(){
 const sl=document.getElementById("tslide");
 tlbox.classList.toggle("off",sl.disabled);
 const mx=+sl.max||10,v=+sl.value||0,f=Math.min(1,v/mx);
 tlfill.style.width=(f*100)+"%";tlhead.style.left=(f*100)+"%";
 tlcur.textContent=tfmt(v);tldur.textContent=tfmt(mx);
 const tk=document.getElementById("tlticks");
 if(tk.childElementCount!==9){tk.innerHTML="";for(let i=1;i<10;i++){
  const d=document.createElement("div");d.className="tick";
  d.style.left=(i*10)+"%";tk.appendChild(d);}}
}
function tlSeek(ev,final){
 const sl=document.getElementById("tslide");
 if(sl.disabled)return;
 const r=tltrack.getBoundingClientRect();
 const f=Math.min(1,Math.max(0,(ev.clientX-r.left)/r.width));
 sl.value=(f*(+sl.max||10)).toFixed(2);
 tlSync();
 if(final){clearTimeout(tlThrottle);tlThrottle=null;scrub();}
 else if(!tlThrottle)tlThrottle=setTimeout(()=>{tlThrottle=null;scrub();},160);
}
tltrack.addEventListener("pointerdown",e=>{tlDrag=true;
 tltrack.setPointerCapture(e.pointerId);tlSeek(e,false);});
tltrack.addEventListener("pointermove",e=>{
 const sl=document.getElementById("tslide");
 const r=tltrack.getBoundingClientRect();
 const f=Math.min(1,Math.max(0,(e.clientX-r.left)/r.width));
 tlbub.style.left=(f*100)+"%";tlbub.textContent=tfmt(f*(+sl.max||10));
 if(!sl.disabled)tlbub.style.display="block";
 if(tlDrag)tlSeek(e,false);});
tltrack.addEventListener("pointerup",e=>{tlDrag=false;tlSeek(e,true);});
tltrack.addEventListener("pointerleave",()=>{tlbub.style.display="none";});
function scrub(){
 tlSync();
 const t=document.getElementById("tslide").value;
 if(BGMODE==="video"&&!CURJOB){bgv.currentTime=+t;return;}
 if(!CURJOB)return;
 setBgMode("img");
 bg.src=`api/jobs/${CURJOB}/frame_at?t=${t}`;
}

gridBg();chips();load();jobs();setMode('draw');tlSync();
</script></body></html>"""
