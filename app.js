/* ================================================
   MOODIFY — COLORFUL APP LOGIC
   Push-based live sync (no polling)
   ================================================ */

let songrun = false;
let mod = 1;
let songs = [];
let sname = [];
let bool = [];
let mood = [[], [], [], []];
let eqc = 0;
let queueItems = [];
let currentSongIndex = -1;
let isPlaying = false;
let favorites = new Set();
let viewMode = 'grid';

let _lastDetectedMood = null;
let _lastDetectedConfidence = 0;

// ── Sync state ──
let _syncLock = false;

window.onload = async () => {
    initAudio();
    initDragDrop();
    initKeys();
    await loadSongs();       // initial load from server
    // NO setInterval — watcher pushes updates automatically
};


// ┌──────────────────────────────────────────────┐
// │  RECEIVE PUSH FROM PYTHON WATCHER             │
// │  This replaces setInterval polling entirely   │
// └──────────────────────────────────────────────┘
eel.expose(onLibraryChanged);
function onLibraryChanged(data) {
    console.log('[Moodify] 🔄 Library change pushed from backend');
    updateSongsData(data, true);   // silent = true
}


// ===== LOAD (manual / initial) =====
async function loadSongs(silent = false) {
    if (_syncLock) return;
    _syncLock = true;

    try {
        if (!silent) toast('Loading your library...', 'info');
        let data = await eel.get_all_songs()();
        updateSongsData(data, silent);
    } catch (e) {
        console.error('[Moodify] loadSongs error:', e);
        if (!silent) toast('Failed to load songs', 'error');
    } finally {
        _syncLock = false;
    }
}


// ===== SHARED UPDATE LOGIC =====
function updateSongsData(data, silent) {
    if (!data || !Array.isArray(data)) {
        console.warn('[Moodify] Invalid song data received');
        return;
    }

    // ── Fingerprint check: skip if nothing changed ──
    const newFP = data.length + '::' +
        data.map(s => `${s.filename}|${s.cover}`).sort().join('|');
    const oldFP = songs.length + '::' +
        songs.map(s => `${s.filename}|${s.cover}`).sort().join('|');

    if (songs.length > 0 && newFP === oldFP) {
        return;  // no change
    }

    console.log(`[Moodify] Updating: ${songs.length} → ${data.length} songs`);

    // ── Preserve current playing song by filename ──
    const currentFile = (currentSongIndex >= 0 && songs[currentSongIndex])
        ? songs[currentSongIndex].filename
        : null;

    // ── Preserve queued filenames ──
    const queuedFiles = new Set();
    queueItems.forEach(q => {
        if (songs[q.si]) queuedFiles.add(songs[q.si].filename);
    });

    // ── Preserve favorite filenames ──
    const favFiles = new Set();
    favorites.forEach(idx => {
        if (songs[idx]) favFiles.add(songs[idx].filename);
    });

    // ── Save old songs for queue re-mapping ──
    const oldSongs = songs;

    // ── Update data ──
    songs = data;
    sname = [];
    bool = [];
    mood = [[], [], [], []];
    favorites = new Set();

    songs.forEach((s, i) => {
        sname.push(s.title);
        bool.push(queuedFiles.has(s.filename));

        if (s.moods.includes('angry')) mood[0].push(i);
        if (s.moods.includes('happy')) mood[1].push(i);
        if (s.moods.includes('sad')) mood[2].push(i);
        if (s.moods.includes('neutral') || s.moods.includes('surprise'))
            mood[3].push(i);

        // Restore favorites by filename
        if (favFiles.has(s.filename)) favorites.add(i);
    });

    // ── Re-map currentSongIndex ──
    if (currentFile) {
        const newIdx = songs.findIndex(s => s.filename === currentFile);
        if (newIdx >= 0) currentSongIndex = newIdx;
    }

    // ── Re-map queue indices ──
    queueItems = queueItems.map(q => {
        const oldFile = oldSongs[q.si]?.filename;
        if (!oldFile) return null;
        const newIdx = songs.findIndex(s => s.filename === oldFile);
        if (newIdx < 0) return null;
        return { ...q, si: newIdx };
    }).filter(Boolean);

    // ── Re-render UI ──
    renderGrid();
    renderList();

    // ── Restore now-playing highlight ──
    if (currentSongIndex >= 0) {
        const gc = document.getElementById(`sc-${currentSongIndex}`);
        const lc = document.getElementById(`sl-${currentSongIndex}`);
        if (gc) gc.classList.add('now-playing');
        if (lc) lc.classList.add('now-playing');
    }

    document.getElementById('statSongs').textContent = songs.length;
    updCount();

    if (!silent) toast(`${songs.length} songs loaded ✨`, 'success');
}


// ===== HELPERS =====
function emojiFor(m) {
    return { happy: '😊', sad: '😢', angry: '🔥', neutral: '🍃', surprise: '⚡' }[m] || '🎵';
}

function setLastDetectedMood(mood, confidence = 0) {
    _lastDetectedMood = mood;
    _lastDetectedConfidence = confidence;

    const pill = document.getElementById('moodPill');
    const label = document.getElementById('lastMoodLabel');
    if (!pill || !label) return;

    pill.style.display = '';
    pill.classList.remove('happy', 'sad', 'angry', 'neutral', 'surprise');
    if (mood) pill.classList.add(mood);

    const pct = Math.round(Math.max(0, Math.min(1, Number(confidence || 0))) * 100);
    label.textContent = `MOOD: ${(mood || '--').toUpperCase()}${mood ? ` (${pct}%)` : ''}`;
}

function plain(html) {
    const d = document.createElement('div');
    d.innerHTML = html;
    return d.textContent || '';
}


// ===== RENDER =====
function renderGrid() {
    const g = document.getElementById('songsGrid');
    g.innerHTML = '';
    songs.forEach((s, i) => {
        const c = document.createElement('div');
        c.className = 'song-card';
        c.id = `sc-${i}`;
        c.dataset.moods = s.moods.join(',');
        c.dataset.title = s.title.toLowerCase();
        c.style.animationDelay = `${i * 0.04}s`;
        const m = s.moods[0] || '';
        c.innerHTML = `
            <div class="sc-cover">
                <img src="${s.cover}" alt="${s.title}" loading="lazy"
                    onerror="this.src='data:image/svg+xml,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 200 200%22%3E%3Crect fill=%22%23f0f0f0%22 width=%22200%22 height=%22200%22 rx=%2216%22/%3E%3Ctext fill=%22%23ccc%22 x=%2250%25%22 y=%2250%25%22 text-anchor=%22middle%22 dy=%22.35em%22 font-size=%2248%22%3E%3C/text%3E%3C/svg%3E'">
                <div class="sc-mood-stripe ${m}"></div>
                <span class="sc-mood-tag">${emojiFor(m)} ${m}</span>
                <button class="sc-add-btn" onclick="event.stopPropagation();addq(${i})" title="Add to queue"><i class="fas fa-plus"></i></button>
                <button class="sc-play-btn" onclick="event.stopPropagation();playSong(${i})" title="Play"><i class="fas fa-play"></i></button>
            </div>
            <div class="sc-title">${s.title}</div>
            <div class="sc-desc">${plain(s.description).substring(0, 50)}</div>
        `;
        c.onclick = () => playSong(i);
        g.appendChild(c);
    });
}

function renderList() {
    const l = document.getElementById('songsList');
    l.innerHTML = '';
    songs.forEach((s, i) => {
        const item = document.createElement('div');
        item.className = 'sl-item';
        item.id = `sl-${i}`;
        item.dataset.moods = s.moods.join(',');
        item.dataset.title = s.title.toLowerCase();
        const m = s.moods[0] || '';
        item.innerHTML = `
            <span class="sl-num">${i + 1}</span>
            <div class="sl-cover"><img src="${s.cover}" alt=""></div>
            <div class="sl-info">
                <div class="sl-title">${s.title}</div>
                <div class="sl-artist">${plain(s.description).substring(0, 40)}</div>
            </div>
            <div class="sl-mood">${emojiFor(m)} ${m}</div>
            <div class="sl-actions">
                <button class="sl-btn" onclick="event.stopPropagation();addq(${i})"><i class="fas fa-plus"></i></button>
                <button class="sl-btn" onclick="event.stopPropagation();playSong(${i})"><i class="fas fa-play"></i></button>
            </div>
        `;
        item.onclick = () => playSong(i);
        l.appendChild(item);
    });
}

function setView(v, btn) {
    viewMode = v;
    document.querySelectorAll('.vt').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('songsGrid').style.display = v === 'grid' ? '' : 'none';
    document.getElementById('songsList').style.display = v === 'list' ? '' : 'none';
}


// ===== PLAY =====
function playSong(i) {
    if (i < 0 || i >= songs.length) return;
    const s = songs[i];
    currentSongIndex = i;

    document.getElementById('sel').src = encodeURI(s.filename);
    const audio = document.getElementById('main_slider');
    audio.load(); audio.play();

    document.getElementById('pbTitle').textContent = s.title;
    document.getElementById('pbArtist').textContent = plain(s.description).substring(0, 60);
    document.getElementById('pbCoverImg').src = s.cover;

    updateFaviconMood(s.moods[0]);
    document.title = `${s.title} — Moodify 🎵`;

    const lb = document.getElementById('pbLike');
    if (favorites.has(i)) { lb.classList.add('active'); lb.innerHTML = '<i class="fas fa-heart"></i>'; }
    else { lb.classList.remove('active'); lb.innerHTML = '<i class="far fa-heart"></i>'; }

    document.querySelectorAll('.song-card').forEach(c => c.classList.remove('now-playing'));
    document.querySelectorAll('.sl-item').forEach(c => c.classList.remove('now-playing'));
    const gc = document.getElementById(`sc-${i}`);
    const lc = document.getElementById(`sl-${i}`);
    if (gc) gc.classList.add('now-playing');
    if (lc) lc.classList.add('now-playing');

    updateQueueNow(s);
    setPlay(true);
    songrun = true;
}

function updateFaviconMood(mood) {
    const colors = {
        happy: ['%23fbbf24', '%23f97316'],
        sad: ['%233b82f6', '%23667eea'],
        angry: ['%23ef4444', '%23f97316'],
        neutral: ['%2310b981', '%2306b6d4'],
        surprise: ['%238b5cf6', '%23f093fb']
    };
    const c = colors[mood] || ['%231DB954', '%23667eea'];
    const svg = `data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Cdefs%3E%3ClinearGradient id='g' x1='0' y1='0' x2='64' y2='64' gradientUnits='userSpaceOnUse'%3E%3Cstop offset='0%25' stop-color='${c[0]}'/%3E%3Cstop offset='100%25' stop-color='${c[1]}'/%3E%3C/linearGradient%3E%3C/defs%3E%3Ccircle cx='32' cy='32' r='30' fill='url(%23g)'/%3E%3Cpath d='M18 42c8-5 18-5 28 0' stroke='white' stroke-width='3.5' stroke-linecap='round' fill='none'/%3E%3Cpath d='M21 36c6-4 14-4 22 0' stroke='white' stroke-width='3' stroke-linecap='round' fill='none'/%3E%3Cpath d='M25 30c4-3 10-3 14 0' stroke='white' stroke-width='2.5' stroke-linecap='round' fill='none'/%3E%3C/svg%3E`;
    let link = document.querySelector("link[rel*='icon']");
    if (link) link.href = svg;
}

function setPlay(playing) {
    isPlaying = playing;
    document.getElementById('mainPlayIcon').className = playing ? 'fas fa-pause' : 'fas fa-play';
    document.getElementById('playerBar').classList.toggle('playing', playing);
}

function togglePlay() {
    const a = document.getElementById('main_slider');
    if (a.paused) { a.play(); setPlay(true); }
    else { a.pause(); setPlay(false); }
}

function prevSong() {
    if (currentSongIndex > 0) playSong(currentSongIndex - 1);
    else if (songs.length) playSong(songs.length - 1);
}


// ===== QUEUE =====
function addq(i) {
    if (!songrun) { playSong(i); return; }
    if (bool[i]) { toast('Already in queue', 'warning'); return; }

    bool[i] = true;
    const s = songs[i];
    const id = eqc++;

    const el = document.createElement('div');
    el.className = 'qd-item';
    el.id = `qi-${id}`;
    el.innerHTML = `
        <span class="qd-item-num">${queueItems.length + 1}</span>
        <div class="qd-item-cover"><img src="${s.cover}" alt=""></div>
        <div class="qd-item-info">
            <div class="qd-item-title">${s.title}</div>
            <div class="qd-item-mood">${s.moods.map(m => emojiFor(m)).join(' ')} ${s.moods.join(', ')}</div>
        </div>
        <button class="qd-item-x" onclick="rmQ(${id},${i})"><i class="fas fa-xmark"></i></button>
    `;

    document.getElementById('qdList').appendChild(el);
    document.getElementById('qdEmpty').style.display = 'none';
    document.getElementById('qdNextTitle').style.display = '';

    queueItems.push({ id, si: i });
    updBadge();
    toast(`"${s.title}" added to queue`, 'success');
}

function rmQ(id, si) {
    const el = document.getElementById(`qi-${id}`);
    if (el) {
        el.style.opacity = '0';
        el.style.transform = 'translateX(20px)';
        setTimeout(() => {
            el.remove();
            bool[si] = false;
            queueItems = queueItems.filter(q => q.id !== id);
            updBadge();
            renum();
            if (!queueItems.length) {
                document.getElementById('qdEmpty').style.display = '';
                document.getElementById('qdNextTitle').style.display = 'none';
            }
        }, 250);
    }
}

function nextsong() {
    if (!queueItems.length) { toast('Queue is empty', 'warning'); return; }
    const next = queueItems.shift();
    const el = document.getElementById(`qi-${next.id}`);
    bool[next.si] = false;
    playSong(next.si);
    if (el) { el.style.opacity = '0'; el.style.transform = 'translateX(20px)'; setTimeout(() => el.remove(), 250); }
    updBadge(); renum();
    if (!queueItems.length) {
        document.getElementById('qdEmpty').style.display = '';
        document.getElementById('qdNextTitle').style.display = 'none';
    }
}

function renum() {
    document.querySelectorAll('#qdList .qd-item').forEach((el, idx) => {
        const n = el.querySelector('.qd-item-num');
        if (n) n.textContent = idx + 1;
    });
}

function updBadge() {
    const c = queueItems.length;
    ['sidebarCounter', 'topBadge'].forEach(id => {
        const el = document.getElementById(id);
        el.textContent = c;
        el.classList.toggle('show', c > 0);
    });
}

function updateQueueNow(s) {
    document.getElementById('qdNow').style.display = '';
    document.getElementById('qdNowCard').innerHTML = `
        <img src="${s.cover}" alt="">
        <div class="qd-now-info">
            <div class="qd-now-title">${s.title}</div>
            <div class="qd-now-sub">${plain(s.description).substring(0, 40)}</div>
        </div>
        <div class="qd-wave"><span></span><span></span><span></span><span></span></div>
    `;
}

function toggleQueue() {
    document.getElementById('queueDrawer').classList.toggle('open');
    document.getElementById('queueDim').classList.toggle('open');
}


// ===== MODES =====
function setMode(m) {
    mod = m;
    ['modeQ', 'modeE', 'modeR'].forEach(id => document.getElementById(id).classList.remove('mode-on'));
    ({ 1: 'modeQ', 2: 'modeE', 3: 'modeR' })[m] && document.getElementById({ 1: 'modeQ', 2: 'modeE', 3: 'modeR' }[m]).classList.add('mode-on');
    toast(`${['', 'Queue', 'Emotion AI', 'Shuffle'][m]} mode activated`, 'info');
    if (m === 2) getTime();
    if (m === 3) rand_play();
}


// ===== FILTERS =====
function filterByMood(val) {
    document.querySelectorAll('.chip').forEach(c => c.classList.remove('active'));
    let count = 0;
    const sel = viewMode === 'grid' ? '.song-card' : '.sl-item';
    document.querySelectorAll(sel).forEach(card => {
        const moods = card.dataset.moods || '';
        if (val === 'all' || val === 'favorites' || moods.includes(val)) {
            card.classList.remove('hidden'); card.style.display = ''; count++;
        } else {
            card.classList.add('hidden'); card.style.display = 'none';
        }
    });
    document.getElementById('emptyBox').style.display = count ? 'none' : '';
    updCount(count);
}

function chipFilter(el, val) {
    document.querySelectorAll('.chip').forEach(c => c.classList.remove('active'));
    el.classList.add('active');
    filterByMood(val);
}

function filterSongs(q) {
    q = q.toLowerCase().trim();
    let count = 0;
    const sel = viewMode === 'grid' ? '.song-card' : '.sl-item';
    document.querySelectorAll(sel).forEach(card => {
        const t = card.dataset.title || '';
        const m = card.dataset.moods || '';
        if (t.includes(q) || m.includes(q)) {
            card.classList.remove('hidden'); card.style.display = ''; count++;
        } else {
            card.classList.add('hidden'); card.style.display = 'none';
        }
    });
    document.getElementById('emptyBox').style.display = count ? 'none' : '';
    updCount(count);
}

function updCount(c) {
    const n = c !== undefined ? c : songs.length;
    document.getElementById('songCount').textContent = `${n} song${n !== 1 ? 's' : ''}`;
}

// ===== EMOTION / RANDOM =====
let _lastFrameAt = 0;
let _frameThrottleMs = 80;
eel.expose(updateFrame);
function updateFrame(data) {
    const now = performance.now();
    if (now - _lastFrameAt < _frameThrottleMs) return;
    _lastFrameAt = now;
    const img = document.getElementById('camFrame');
    if (img) img.src = data;
}

let _emotionPollTimer = null;
async function _startEmotionStatusPoll() {
    if (_emotionPollTimer) return;
    const moodEl = document.getElementById('detectedMood');
    _emotionPollTimer = setInterval(async () => {
        try {
            const s = await eel.getCameraStatus()();
            if (!s) return;
            const emo = (s.emotion || 'neutral');
            const conf = Math.max(0, Math.min(1, Number(s.confidence || 0)));
            if (moodEl) moodEl.textContent = `${emo.toUpperCase()} ${Math.round(conf * 100)}%`;
            setLastDetectedMood(emo, conf);
        } catch (e) {
            // ignore
        }
    }, 250);
}

function _stopEmotionStatusPoll() {
    if (_emotionPollTimer) {
        clearInterval(_emotionPollTimer);
        _emotionPollTimer = null;
    }
}

function openEmotionModal() {
    document.getElementById('emotionModal').classList.add('open');
    document.getElementById('detectedMood').textContent = 'Analyzing...';
    eel.startCamera()();
    _startEmotionStatusPoll();
}

function closeEmotionModal() {
    document.getElementById('emotionModal').classList.remove('open');
    _stopEmotionStatusPoll();
    eel.stopCamera()();
}

async function getTime() {
    try {
        openEmotionModal();
        await new Promise(r => setTimeout(r, 1200));
        const s = await eel.getCameraStatus()();
        const val = (s && s.emotion) ? s.emotion : 'neutral';
        const conf = (s && s.confidence) ? s.confidence : 0;
        setLastDetectedMood(val, conf);
        const map = { angry: 0, happy: 1, sad: 2 };
        setTimeout(() => {
            closeEmotionModal();
            moody(map[val] !== undefined ? map[val] : 3);
            toast(`Mood detected: ${val} ${emojiFor(val)}`, 'success');
        }, 3800);
    } catch (e) {
        console.error(e);
        closeEmotionModal();
        toast('Detection failed, shuffling instead', 'warning');
        rand_play();
    }
}

function moody(v) {
    if (!mood[v].length) { rand_play(); return; }
    playSong(mood[v][Math.floor(Math.random() * mood[v].length)]);
}

function rand_play() {
    if (!songs.length) return;
    playSong(Math.floor(Math.random() * songs.length));
}


// ===== AUDIO =====
function initAudio() {
    const a = document.getElementById('main_slider');
    a.volume = 0.8;

    a.addEventListener('timeupdate', () => {
        if (!a.duration) return;
        const p = (a.currentTime / a.duration) * 100;
        // console.log(`[Moodify] Progress: ${p.toFixed(2)}%`);
        document.getElementById('pbFill').style.width = p + '%';
        document.getElementById('timeFill').style.width = p + '%';
        document.getElementById('curTime').textContent = fmt(a.currentTime);
        document.getElementById('totTime').textContent = fmt(a.duration);
    });

    a.addEventListener('ended', () => {
        setPlay(false);
        if (mod === 1) { queueItems.length ? nextsong() : toast('Queue finished', 'info'); }
        else if (mod === 2) getTime();
        else rand_play();
    });

    a.addEventListener('play', () => setPlay(true));
    a.addEventListener('pause', () => setPlay(false));
}

function fmt(s) {
    const m = Math.floor(s / 60);
    const sec = Math.floor(s % 60);
    return `${m}:${sec.toString().padStart(2, '0')}`;
}

function seek(e) {
    const bar = e.currentTarget;
    const rect = bar.getBoundingClientRect();
    const pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    const a = document.getElementById('main_slider');
    if (a.duration) a.currentTime = pct * a.duration;
}

function setVolume(v) {
    document.getElementById('main_slider').volume = v / 100;
    const ic = document.getElementById('volIcon');
    ic.className = v == 0 ? 'fas fa-volume-xmark' : v < 50 ? 'fas fa-volume-low' : 'fas fa-volume-high';
}

function toggleMute() {
    const a = document.getElementById('main_slider');
    const r = document.getElementById('volRange');
    if (a.volume > 0) { a._pv = a.volume; a.volume = 0; r.value = 0; }
    else { a.volume = a._pv || 0.8; r.value = a.volume * 100; }
    setVolume(r.value);
}


// ===== LIKE =====
function toggleLike() {
    if (currentSongIndex < 0) return;
    const b = document.getElementById('pbLike');
    if (favorites.has(currentSongIndex)) {
        favorites.delete(currentSongIndex);
        b.classList.remove('active');
        b.innerHTML = '<i class="far fa-heart"></i>';
        toast('Removed from favorites', 'info');
    } else {
        favorites.add(currentSongIndex);
        b.classList.add('active');
        b.innerHTML = '<i class="fas fa-heart"></i>';
        toast('Added to favorites ❤️', 'success');
    }
}


// ===== VIEW =====
function showView(v, el) {
    document.querySelectorAll('.sidebar-nav .nav-link').forEach(n => n.classList.remove('active'));
    if (el) el.classList.add('active');
    document.getElementById('heroSection').style.display = v === 'home' ? '' : 'none';
}


// ===== UPLOAD =====
function openUploadModal() { document.getElementById('uploadModal').classList.add('open'); }
function closeUploadModal() {
    document.getElementById('uploadModal').classList.remove('open');
    document.getElementById('uploadFile').value = '';
    document.getElementById('uploadTitle').value = '';
    document.getElementById('uploadDesc').value = '';
    document.getElementById('dzFilename').textContent = '';
    document.getElementById('uploadLoading').style.display = 'none';
}

function onFileChosen(inp) {
    if (inp.files.length) {
        document.getElementById('dzFilename').textContent = `✓ ${inp.files[0].name}`;
        if (!document.getElementById('uploadTitle').value)
            document.getElementById('uploadTitle').value = inp.files[0].name.replace('.mp3', '');
    }
}

async function handleUpload() {
    const fi = document.getElementById('uploadFile');
    if (!fi.files.length) { toast('Select a file first!', 'error'); return; }
    const file = fi.files[0];
    const reader = new FileReader();
    reader.onload = async (e) => {
        const meta = {
            title: document.getElementById('uploadTitle').value || file.name.replace('.mp3', ''),
            description: document.getElementById('uploadDesc').value || 'Uploaded Song',
            moods: [document.getElementById('uploadMood').value],
            cover_b64: ''
        };
        try {
            document.getElementById('uploadLoading').style.display = '';
            let res = await eel.upload_song(e.target.result, meta)();
            if (res.success) {
                toast('Upload successful! ✨', 'success');
                closeUploadModal();
                // Watcher will auto-push update
                // But also do manual refresh as safety net
                setTimeout(() => loadSongs(true), 2000);
            }
            else toast('Upload failed: ' + res.message, 'error');
        } catch (err) { console.error(err); toast('Upload error', 'error'); }
        finally { document.getElementById('uploadLoading').style.display = 'none'; }
    };
    reader.readAsDataURL(file);
}

function switchToYouTube() {
    closeUploadModal();
    setTimeout(() => openYouTubeModal(), 200);
}


// ===== YOUTUBE DOWNLOAD =====
function openYouTubeModal() {
    document.getElementById('ytModal').classList.add('open');
    resetYtStep();
}
function closeYouTubeModal() {
    document.getElementById('ytModal').classList.remove('open');
    resetYtStep();
}

function resetYtStep() {
    // Full reset (including input)
    resetYtUI();

    // Clear input
    const urlInput = document.getElementById('ytUrl');
    if (urlInput) {
        urlInput.value = '';
        urlInput.style.borderColor = '';
    }
}

function resetYtUI() {
    // Partial reset (keeps input value) - useful for oninput
    // Ensure the main body is visible and loading is hidden
    document.querySelector('#ytModal .modal-body').style.display = 'block';

    // Reset specific loading UI elements if they exist
    const loadingEl = document.getElementById('ytLoading');
    if (loadingEl) loadingEl.style.display = 'none';

    const pBar = document.getElementById('ytProgressBar');
    if (pBar) pBar.style.width = '0%';

    const pText = document.getElementById('ytProgressText');
    if (pText) pText.textContent = '0%';

    const pSpeed = document.getElementById('ytSpeed');
    if (pSpeed) pSpeed.textContent = '';

    document.getElementById('ytStep1').style.display = 'block';
    document.getElementById('ytPreview').style.display = 'none';
    document.getElementById('ytMoodSection').style.display = 'none';
    document.getElementById('ytDownloadBtn').style.display = 'none';

    // Clear preview
    const thumb = document.getElementById('ytThumb');
    if (thumb) thumb.src = '';

    const title = document.getElementById('ytTitle');
    if (title) title.textContent = '';

    const author = document.getElementById('ytAuthor');
    if (author) author.textContent = '';
}

eel.expose(updateDownloadProgress);
function updateDownloadProgress(data) {
    // data = { percent: 12.5, speed: "2.5MiB/s", eta: "00:30", status: "downloading" }
    const loadingEl = document.getElementById('ytLoading');
    if (loadingEl && loadingEl.style.display !== 'none') {
        const pBar = document.getElementById('ytProgressBar');
        if (pBar) pBar.style.width = data.percent + '%';

        const pText = document.getElementById('ytProgressText');
        if (pText) pText.textContent = Math.round(data.percent) + '%';

        const pSpeed = document.getElementById('ytSpeed');
        if (pSpeed) {
            if (data.status === 'finished') {
                pSpeed.textContent = 'Converting...';
            } else {
                pSpeed.textContent = `${data.speed} • ETA: ${data.eta}`;
            }
        }
    }
}

async function checkYouTubeUrl() {
    const url = document.getElementById('ytUrl').value.trim();
    if (!url) return;

    const btn = document.querySelector('.btn-check');
    const originalIcon = btn.innerHTML;

    try {
        btn.innerHTML = '<div class="spinner" style="width:16px;height:16px;border-width:2px;border-color:#666;border-top-color:transparent"></div>';

        // Call backend to get info
        let res = await eel.get_youtube_info(url)();

        if (res.success) {
            const info = res.data;
            document.getElementById('ytThumb').src = info.thumbnail;
            document.getElementById('ytTitle').textContent = info.title;
            document.getElementById('ytAuthor').textContent = info.author;

            // Show next steps
            document.getElementById('ytPreview').style.display = 'flex';
            document.getElementById('ytMoodSection').style.display = 'block';
            document.getElementById('ytDownloadBtn').style.display = 'block';
        } else {
            toast('Could not find video', 'error');
            document.getElementById('ytUrl').style.borderColor = 'red';
        }
    } catch (e) {
        console.error(e);
        toast('Connection error', 'error');
    } finally {
        btn.innerHTML = originalIcon;
    }
}

async function submitYouTube() {
    const url = document.getElementById('ytUrl').value.trim();
    const mood = document.getElementById('ytMood').value;

    if (!url) { toast('Please enter a valid URL', 'error'); return; }

    try {
        // Switch to full loading state
        document.querySelector('#ytModal .modal-body').style.display = 'none';
        document.getElementById('ytLoading').style.display = 'flex';
        document.getElementById('ytLoadingText').textContent = 'Downloading & Converting...';

        let res = await eel.download_youtube_song(url, mood)();

        if (res.success) {
            toast('Download complete! 🎉', 'success');
            closeYouTubeModal();
            setTimeout(() => loadSongs(true), 1500);
        } else {
            toast('Download failed: ' + res.message, 'error');
            // Restore UI
            document.querySelector('#ytModal .modal-body').style.display = 'block';
            document.getElementById('ytLoading').style.display = 'none';
        }
    } catch (e) {
        console.error(e);
        toast('Error interacting with backend', 'error');
        document.querySelector('#ytModal .modal-body').style.display = 'block';
        document.getElementById('ytLoading').style.display = 'none';
    }
}








// ===== DRAG & DROP =====
function initDragDrop() {
    const dz = document.getElementById('dropZone');
    if (!dz) return;
    ['dragenter', 'dragover'].forEach(ev => dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.add('dragover'); }));
    ['dragleave', 'drop'].forEach(ev => dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.remove('dragover'); }));
    dz.addEventListener('drop', e => {
        if (e.dataTransfer.files.length) {
            document.getElementById('uploadFile').files = e.dataTransfer.files;
            onFileChosen(document.getElementById('uploadFile'));
        }
    });
}


// ===== KEYBOARD =====
function initKeys() {
    document.addEventListener('keydown', e => {
        if (['INPUT', 'SELECT', 'TEXTAREA'].includes(e.target.tagName)) return;
        switch (e.code) {
            case 'Space': e.preventDefault(); togglePlay(); break;
            case 'ArrowRight': nextsong(); break;
            case 'ArrowLeft': prevSong(); break;
            case 'KeyQ': toggleQueue(); break;
        }
        if ((e.metaKey || e.ctrlKey) && e.code === 'KeyK') {
            e.preventDefault();
            document.getElementById('searchInput').focus();
        }
    });
}


// ===== TOAST =====
function toast(msg, type = 'info') {
    const c = document.getElementById('toastStack');
    const t = document.createElement('div');
    t.className = `toast-item ${type}`;
    const icons = { success: 'fa-circle-check', error: 'fa-circle-xmark', warning: 'fa-triangle-exclamation', info: 'fa-circle-info' };
    t.innerHTML = `<i class="fas ${icons[type] || icons.info}"></i><span>${msg}</span>`;
    c.appendChild(t);
    setTimeout(() => t.remove(), 4000);
}

// ===== SYSTEM =====
eel.expose(close_app);
function close_app() {
    window.close();
}

// Notify backend when window closes
window.onbeforeunload = function () {
    // This is optional as Eel usually handles this, but good for safety
    // We can't really call eel functions reliably here as the window is closing
    // but in some setups it helps. 
};