/* ================================================
   MOODIFY — PREMIUM APP LOGIC v2.0
   Push-based live sync + Visualizer + Theme + Premium UX
   ================================================ */

// ── App State ──
const APP = {
    songrun: false,
    mod: 1,
    songs: [],
    sname: [],
    bool: [],
    mood: [[], [], [], []],
    eqc: 0,
    queueItems: [],
    currentSongIndex: -1,
    isPlaying: false,
    favorites: new Set(),
    viewMode: 'grid',
    repeatMode: 0, // 0=off, 1=all, 2=one
    playCount: 0,
    recentlyPlayed: [],
    maxRecent: 20,
    _lastDetectedMood: null,
    _lastDetectedConfidence: 0,
    _syncLock: false,
    _lastFrameAt: 0,
    _frameThrottleMs: 80,
    _emotionPollTimer: null,
    sortMode: 'default',
    theme: 'light',
};

// ── Shorthand getters ──
let get = (id) => document.getElementById(id);
let qs = (s) => document.querySelector(s);
let qsa = (s) => document.querySelectorAll(s);

// Backward-compat aliases
let songrun, mod, songs, sname, bool, mood, eqc, queueItems,
    currentSongIndex, isPlaying, favorites, viewMode;

function syncAliases() {
    songrun = APP.songrun;
    mod = APP.mod;
    songs = APP.songs;
    sname = APP.sname;
    bool = APP.bool;
    mood = APP.mood;
    eqc = APP.eqc;
    queueItems = APP.queueItems;
    currentSongIndex = APP.currentSongIndex;
    isPlaying = APP.isPlaying;
    favorites = APP.favorites;
    viewMode = APP.viewMode;
}

// ┌──────────────────────────────────────────────┐
// │  INITIALIZATION                               │
// └──────────────────────────────────────────────┘
window.onload = async () => {
    loadPreferences();
    initTheme();
    initAudio();
    initVisualizer();
    initDragDrop();
    initKeys();
    initScrollAnimations();
    initParticles();
    initTopbarScroll();
    showSkeletons(true);
    await loadSongs();
    showSkeletons(false);
    renderRecentlyPlayed();
    animateCounters();
    syncAliases();
};

// ┌──────────────────────────────────────────────┐
// │  THEME SYSTEM                                 │
// └──────────────────────────────────────────────┘
function initTheme() {
    const saved = localStorage.getItem('moodify-theme');
    if (saved) {
        APP.theme = saved;
    } else {
        // Auto-detect system preference
        const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
        APP.theme = prefersDark ? 'dark' : 'light';
    }
    applyTheme(APP.theme);
}

function toggleTheme() {
    APP.theme = APP.theme === 'light' ? 'dark' : 'light';
    applyTheme(APP.theme);
    localStorage.setItem('moodify-theme', APP.theme);
    toast(`${APP.theme === 'dark' ? '🌙 Dark' : '☀️ Light'} mode activated`, 'info');
}

function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    const icon = get('themeIcon');
    if (icon) icon.className = theme === 'dark' ? 'fas fa-sun' : 'fas fa-moon';
    const meta = get('metaThemeColor');
    if (meta) meta.content = theme === 'dark' ? '#0a0a1a' : '#ffffff';
    const sdVal = get('sdThemeVal');
    if (sdVal) sdVal.textContent = theme === 'dark' ? 'Dark' : 'Light';
}

// ┌──────────────────────────────────────────────┐
// │  PREFERENCES (localStorage)                   │
// └──────────────────────────────────────────────┘
function loadPreferences() {
    try {
        const favs = JSON.parse(localStorage.getItem('moodify-favorites') || '[]');
        APP.favorites = new Set(favs);
        const recent = JSON.parse(localStorage.getItem('moodify-recent') || '[]');
        APP.recentlyPlayed = recent;
        const vol = localStorage.getItem('moodify-volume');
        if (vol !== null) {
            const r = get('volRange');
            if (r) r.value = vol;
        }
        APP.playCount = parseInt(localStorage.getItem('moodify-playcount') || '0');
    } catch (e) {
        console.warn('[Moodify] Prefs load error:', e);
    }
}

function savePreferences() {
    try {
        localStorage.setItem('moodify-favorites', JSON.stringify([...APP.favorites]));
        localStorage.setItem('moodify-recent', JSON.stringify(APP.recentlyPlayed));
        localStorage.setItem('moodify-volume', get('volRange')?.value || '80');
        localStorage.setItem('moodify-playcount', String(APP.playCount));
    } catch (e) { /* quota exceeded? */ }
}

function resetPreferences() {
    localStorage.clear();
    APP.favorites = new Set();
    APP.recentlyPlayed = [];
    APP.playCount = 0;
    applyTheme('light');
    toast('Preferences reset', 'info');
}

// ┌──────────────────────────────────────────────┐
// │  PUSH SYNC FROM PYTHON WATCHER                │
// └──────────────────────────────────────────────┘
eel.expose(onLibraryChanged);
function onLibraryChanged(data) {
    console.log('[Moodify] 🔄 Library change pushed from backend');
    updateSongsData(data, true);
}

// ===== LOAD (manual / initial) =====
async function loadSongs(silent = false) {
    if (APP._syncLock) return;
    APP._syncLock = true;
    try {
        if (!silent) toast('Loading your library...', 'info');
        let data = await eel.get_all_songs()();
        updateSongsData(data, silent);
    } catch (e) {
        console.error('[Moodify] loadSongs error:', e);
        if (!silent) toast('Failed to load songs', 'error');
    } finally {
        APP._syncLock = false;
    }
}

// ===== SHARED UPDATE LOGIC =====
function updateSongsData(data, silent) {
    if (!data || !Array.isArray(data)) {
        console.warn('[Moodify] Invalid song data');
        return;
    }

    const newFP = data.length + '::' + data.map(s => `${s.filename}|${s.cover}`).sort().join('|');
    const oldFP = APP.songs.length + '::' + APP.songs.map(s => `${s.filename}|${s.cover}`).sort().join('|');
    if (APP.songs.length > 0 && newFP === oldFP) return;

    console.log(`[Moodify] Updating: ${APP.songs.length} → ${data.length} songs`);

    const currentFile = (APP.currentSongIndex >= 0 && APP.songs[APP.currentSongIndex])
        ? APP.songs[APP.currentSongIndex].filename : null;

    const queuedFiles = new Set();
    APP.queueItems.forEach(q => {
        if (APP.songs[q.si]) queuedFiles.add(APP.songs[q.si].filename);
    });

    const favFiles = new Set();
    APP.favorites.forEach(idx => {
        if (APP.songs[idx]) favFiles.add(APP.songs[idx].filename);
    });

    const oldSongs = APP.songs;
    APP.songs = data;
    APP.sname = [];
    APP.bool = [];
    APP.mood = [[], [], [], []];
    APP.favorites = new Set();

    APP.songs.forEach((s, i) => {
        APP.sname.push(s.title);
        APP.bool.push(queuedFiles.has(s.filename));
        if (s.moods.includes('angry')) APP.mood[0].push(i);
        if (s.moods.includes('happy')) APP.mood[1].push(i);
        if (s.moods.includes('sad')) APP.mood[2].push(i);
        if (s.moods.includes('neutral') || s.moods.includes('surprise')) APP.mood[3].push(i);
        if (favFiles.has(s.filename)) APP.favorites.add(i);
    });

    if (currentFile) {
        const newIdx = APP.songs.findIndex(s => s.filename === currentFile);
        if (newIdx >= 0) APP.currentSongIndex = newIdx;
    }

    APP.queueItems = APP.queueItems.map(q => {
        const oldFile = oldSongs[q.si]?.filename;
        if (!oldFile) return null;
        const newIdx = APP.songs.findIndex(s => s.filename === oldFile);
        if (newIdx < 0) return null;
        return { ...q, si: newIdx };
    }).filter(Boolean);

    syncAliases();
    renderGrid();
    renderList();
    showSkeletons(false);

    if (APP.currentSongIndex >= 0) {
        const gc = get(`sc-${APP.currentSongIndex}`);
        const lc = get(`sl-${APP.currentSongIndex}`);
        if (gc) gc.classList.add('now-playing');
        if (lc) lc.classList.add('now-playing');
    }

    get('statSongs').setAttribute('data-target', APP.songs.length);
    get('statSongs').textContent = APP.songs.length;
    const ssc = get('sidebarSongCount');
    if (ssc) ssc.textContent = APP.songs.length;
    const spc = get('sidebarPlayCount');
    if (spc) spc.textContent = APP.playCount;
    updCount();
    updateFavCounter();

    if (!silent) toast(`${APP.songs.length} songs loaded ✨`, 'success');
    savePreferences();
}

// ┌──────────────────────────────────────────────┐
// │  SKELETON LOADING                             │
// └──────────────────────────────────────────────┘
function showSkeletons(show) {
    const sk = get('skeletonGrid');
    const sg = get('songsGrid');
    const sl = get('songsList');
    if (sk) sk.style.display = show ? '' : 'none';
    if (sg) sg.style.display = show ? 'none' : (APP.viewMode === 'grid' ? '' : 'none');
    if (sl) sl.style.display = show ? 'none' : (APP.viewMode === 'list' ? '' : 'none');
}

// ┌──────────────────────────────────────────────┐
// │  HELPERS                                      │
// └──────────────────────────────────────────────┘
function emojiFor(m) {
    return { happy: '😊', sad: '😢', angry: '🔥', neutral: '🍃', surprise: '⚡' }[m] || '🎵';
}

function colorFor(m) {
    return { happy: '#fbbf24', sad: '#3b82f6', angry: '#ef4444', neutral: '#10b981', surprise: '#8b5cf6' }[m] || '#1DB954';
}

function setLastDetectedMood(mood, confidence = 0) {
    APP._lastDetectedMood = mood;
    APP._lastDetectedConfidence = confidence;
    const pill = get('moodPill');
    const label = get('lastMoodLabel');
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

function debounce(fn, ms) {
    let t;
    return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
}

const debouncedSearch = debounce((val) => filterSongs(val), 250);

// ┌──────────────────────────────────────────────┐
// │  RENDER                                       │
// └──────────────────────────────────────────────┘
function renderGrid() {
    const g = get('songsGrid');
    g.innerHTML = '';
    APP.songs.forEach((s, i) => {
        const c = document.createElement('div');
        c.className = 'song-card fade-in-up';
        c.id = `sc-${i}`;
        c.dataset.moods = s.moods.join(',');
        c.dataset.title = s.title.toLowerCase();
        c.style.animationDelay = `${Math.min(i * 0.04, 1)}s`;
        const m = s.moods[0] || '';
        const isFav = APP.favorites.has(i);
        c.innerHTML = `
            <div class="sc-cover">
                <img src="${s.cover}" alt="${s.title}" loading="lazy"
                    onerror="this.src='data:image/svg+xml,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 200 200%22%3E%3Crect fill=%22%23f0f0f0%22 width=%22200%22 height=%22200%22 rx=%2216%22/%3E%3Ctext fill=%22%23ccc%22 x=%2250%25%22 y=%2250%25%22 text-anchor=%22middle%22 dy=%22.35em%22 font-size=%2248%22%3E🎵%3C/text%3E%3C/svg%3E'">
                <div class="sc-mood-stripe ${m}"></div>
                <span class="sc-mood-tag">${emojiFor(m)} ${m}</span>
                <button class="sc-fav-btn ${isFav ? 'active' : ''}" onclick="event.stopPropagation();toggleFavAt(${i})" title="Favorite">
                    <i class="${isFav ? 'fas' : 'far'} fa-heart"></i>
                </button>
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
    const l = get('songsList');
    l.innerHTML = '';
    APP.songs.forEach((s, i) => {
        const item = document.createElement('div');
        item.className = 'sl-item';
        item.id = `sl-${i}`;
        item.dataset.moods = s.moods.join(',');
        item.dataset.title = s.title.toLowerCase();
        const m = s.moods[0] || '';
        item.innerHTML = `
            <span class="sl-num">${i + 1}</span>
            <div class="sl-cover"><img src="${s.cover}" alt="" loading="lazy"></div>
            <div class="sl-info">
                <div class="sl-title">${s.title}</div>
                <div class="sl-artist">${plain(s.description).substring(0, 40)}</div>
            </div>
            <div class="sl-mood">${emojiFor(m)} ${m}</div>
            <div class="sl-actions">
                <button class="sl-btn" onclick="event.stopPropagation();addq(${i})"><i class="fas fa-plus"></i></button>
                <button class="sl-btn play" onclick="event.stopPropagation();playSong(${i})"><i class="fas fa-play"></i></button>
            </div>
        `;
        item.onclick = () => playSong(i);
        l.appendChild(item);
    });
}

function setView(v, btn) {
    APP.viewMode = v;
    viewMode = v;
    qsa('.vt').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    get('songsGrid').style.display = v === 'grid' ? '' : 'none';
    get('songsList').style.display = v === 'list' ? '' : 'none';
}

// ┌──────────────────────────────────────────────┐
// │  RECENTLY PLAYED                              │
// └──────────────────────────────────────────────┘
function addToRecentlyPlayed(i) {
    if (i < 0 || i >= APP.songs.length) return;
    const filename = APP.songs[i].filename;
    APP.recentlyPlayed = APP.recentlyPlayed.filter(f => f !== filename);
    APP.recentlyPlayed.unshift(filename);
    if (APP.recentlyPlayed.length > APP.maxRecent) APP.recentlyPlayed.pop();
    APP.playCount++;
    const spc = get('sidebarPlayCount');
    if (spc) spc.textContent = APP.playCount;
    savePreferences();
    renderRecentlyPlayed();
}

function renderRecentlyPlayed() {
    const section = get('recentSection');
    const scroll = get('recentScroll');
    if (!scroll) return;

    const recentSongs = APP.recentlyPlayed
        .map(f => APP.songs.find(s => s.filename === f))
        .filter(Boolean)
        .slice(0, 10);

    if (!recentSongs.length) {
        if (section) section.style.display = 'none';
        return;
    }

    if (section) section.style.display = '';
    scroll.innerHTML = '';

    recentSongs.forEach((s) => {
        const idx = APP.songs.indexOf(s);
        const card = document.createElement('div');
        card.className = 'recent-card';
        card.onclick = () => playSong(idx);
        card.innerHTML = `
            <div class="rc-cover">
                <img src="${s.cover}" alt="${s.title}" loading="lazy">
                <div class="rc-play"><i class="fas fa-play"></i></div>
            </div>
            <div class="rc-title">${s.title}</div>
            <div class="rc-mood">${emojiFor(s.moods[0])} ${s.moods[0] || ''}</div>
        `;
        scroll.appendChild(card);
    });
}

function clearRecentlyPlayed() {
    APP.recentlyPlayed = [];
    savePreferences();
    renderRecentlyPlayed();
    toast('History cleared', 'info');
}

// ┌──────────────────────────────────────────────┐
// │  PLAY                                         │
// └──────────────────────────────────────────────┘
function playSong(i) {
    if (i < 0 || i >= APP.songs.length) return;
    const s = APP.songs[i];
    APP.currentSongIndex = i;
    currentSongIndex = i;

    get('sel').src = encodeURI(s.filename);
    const audio = get('main_slider');
    audio.load();
    audio.play();

    // Player bar
    get('pbTitle').textContent = s.title;
    get('pbArtist').textContent = plain(s.description).substring(0, 60);
    get('pbCoverImg').src = s.cover;

    // Fullscreen player
    const fsTitle = get('fsTitle');
    const fsArtist = get('fsArtist');
    const fsCoverImg = get('fsCoverImg');
    const fsMoodTag = get('fsMoodTag');
    const fsBg = get('fsBg');
    if (fsTitle) fsTitle.textContent = s.title;
    if (fsArtist) fsArtist.textContent = plain(s.description).substring(0, 80);
    if (fsCoverImg) fsCoverImg.src = s.cover;
    if (fsMoodTag) fsMoodTag.innerHTML = `${emojiFor(s.moods[0])} ${s.moods[0] || ''}`;
    if (fsBg) fsBg.style.backgroundImage = `url(${s.cover})`;

    updateFaviconMood(s.moods[0]);
    document.title = `${s.title} — Moodify 🎵`;

    // Like state
    updateLikeUI();

    // Highlight current
    qsa('.song-card').forEach(c => c.classList.remove('now-playing'));
    qsa('.sl-item').forEach(c => c.classList.remove('now-playing'));
    const gc = get(`sc-${i}`);
    const lc = get(`sl-${i}`);
    if (gc) { gc.classList.add('now-playing'); gc.scrollIntoView({ behavior: 'smooth', block: 'nearest' }); }
    if (lc) lc.classList.add('now-playing');

    updateQueueNow(s);
    setPlay(true);
    APP.songrun = true;
    songrun = true;

    addToRecentlyPlayed(i);
    connectVisualizer();
}

function updateLikeUI() {
    const isFav = APP.favorites.has(APP.currentSongIndex);
    const lb = get('pbLike');
    if (lb) {
        lb.classList.toggle('active', isFav);
        lb.innerHTML = `<i class="${isFav ? 'fas' : 'far'} fa-heart"></i>`;
    }
    const fsLike = get('fsLikeIcon');
    if (fsLike) fsLike.className = `${isFav ? 'fas' : 'far'} fa-heart`;
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
    APP.isPlaying = playing;
    isPlaying = playing;
    get('mainPlayIcon').className = playing ? 'fas fa-pause' : 'fas fa-play';
    const fsIcon = get('fsPlayIcon');
    if (fsIcon) fsIcon.className = playing ? 'fas fa-pause' : 'fas fa-play';
    get('playerBar').classList.toggle('playing', playing);
}

function togglePlay() {
    const a = get('main_slider');
    if (a.paused) { a.play(); setPlay(true); }
    else { a.pause(); setPlay(false); }
}

function prevSong() {
    const a = get('main_slider');
    if (a.currentTime > 3) { a.currentTime = 0; return; }
    if (APP.currentSongIndex > 0) playSong(APP.currentSongIndex - 1);
    else if (APP.songs.length) playSong(APP.songs.length - 1);
}

function toggleRepeat() {
    APP.repeatMode = (APP.repeatMode + 1) % 3;
    const btn = get('repeatBtn');
    if (!btn) return;
    btn.classList.toggle('active', APP.repeatMode > 0);
    if (APP.repeatMode === 2) btn.innerHTML = '<i class="fas fa-repeat"></i><span class="repeat-one">1</span>';
    else btn.innerHTML = '<i class="fas fa-repeat"></i>';
    const labels = ['Repeat off', 'Repeat all', 'Repeat one'];
    toast(labels[APP.repeatMode], 'info');
}

// ┌──────────────────────────────────────────────┐
// │  QUEUE                                        │
// └──────────────────────────────────────────────┘
function addq(i) {
    if (!APP.songrun) { playSong(i); return; }
    if (APP.bool[i]) { toast('Already in queue', 'warning'); return; }
    APP.bool[i] = true;
    const s = APP.songs[i];
    const id = APP.eqc++;

    const el = document.createElement('div');
    el.className = 'qd-item';
    el.id = `qi-${id}`;
    el.innerHTML = `
        <span class="qd-item-num">${APP.queueItems.length + 1}</span>
        <div class="qd-item-cover"><img src="${s.cover}" alt=""></div>
        <div class="qd-item-info">
            <div class="qd-item-title">${s.title}</div>
            <div class="qd-item-mood">${s.moods.map(m => emojiFor(m)).join(' ')} ${s.moods.join(', ')}</div>
        </div>
        <button class="qd-item-x" onclick="rmQ(${id},${i})"><i class="fas fa-xmark"></i></button>
    `;

    get('qdList').appendChild(el);
    get('qdEmpty').style.display = 'none';
    get('qdNextTitle').style.display = '';
    APP.queueItems.push({ id, si: i });
    syncAliases();
    updBadge();
    toast(`"${s.title}" added to queue`, 'success');
}

function rmQ(id, si) {
    const el = get(`qi-${id}`);
    if (el) {
        el.style.opacity = '0';
        el.style.transform = 'translateX(20px)';
        setTimeout(() => {
            el.remove();
            APP.bool[si] = false;
            APP.queueItems = APP.queueItems.filter(q => q.id !== id);
            syncAliases();
            updBadge();
            renum();
            if (!APP.queueItems.length) {
                get('qdEmpty').style.display = '';
                get('qdNextTitle').style.display = 'none';
            }
        }, 250);
    }
}

function nextsong() {
    if (!APP.queueItems.length) { toast('Queue is empty', 'warning'); return; }
    const next = APP.queueItems.shift();
    const el = get(`qi-${next.id}`);
    APP.bool[next.si] = false;
    playSong(next.si);
    if (el) { el.style.opacity = '0'; el.style.transform = 'translateX(20px)'; setTimeout(() => el.remove(), 250); }
    syncAliases();
    updBadge();
    renum();
    if (!APP.queueItems.length) {
        get('qdEmpty').style.display = '';
        get('qdNextTitle').style.display = 'none';
    }
}

function clearQueue() {
    APP.queueItems.forEach(q => { APP.bool[q.si] = false; });
    APP.queueItems = [];
    syncAliases();
    get('qdList').querySelectorAll('.qd-item').forEach(el => el.remove());
    get('qdEmpty').style.display = '';
    get('qdNextTitle').style.display = 'none';
    updBadge();
    toast('Queue cleared', 'info');
}

function renum() {
    qsa('#qdList .qd-item').forEach((el, idx) => {
        const n = el.querySelector('.qd-item-num');
        if (n) n.textContent = idx + 1;
    });
    const qdCount = get('qdCount');
    if (qdCount) qdCount.textContent = APP.queueItems.length;
}

function updBadge() {
    const c = APP.queueItems.length;
    ['sidebarCounter', 'topBadge'].forEach(id => {
        const el = get(id);
        if (el) { el.textContent = c; el.classList.toggle('show', c > 0); }
    });
    const qdCount = get('qdCount');
    if (qdCount) qdCount.textContent = c;
}

function updateFavCounter() {
    const fc = get('favCounter');
    if (fc) {
        fc.textContent = APP.favorites.size;
        fc.classList.toggle('show', APP.favorites.size > 0);
    }
}

function updateQueueNow(s) {
    get('qdNow').style.display = '';
    get('qdNowCard').innerHTML = `
        <img src="${s.cover}" alt="">
        <div class="qd-now-info">
            <div class="qd-now-title">${s.title}</div>
            <div class="qd-now-sub">${plain(s.description).substring(0, 40)}</div>
        </div>
        <div class="qd-wave"><span></span><span></span><span></span><span></span></div>
    `;
}

function toggleQueue() {
    get('queueDrawer').classList.toggle('open');
    get('queueDim').classList.toggle('open');
}

// ┌──────────────────────────────────────────────┐
// │  MODES                                        │
// └──────────────────────────────────────────────┘
function setMode(m) {
    APP.mod = m;
    mod = m;
    ['modeQ', 'modeE', 'modeR'].forEach(id => get(id).classList.remove('mode-on'));
    const modeMap = { 1: 'modeQ', 2: 'modeE', 3: 'modeR' };
    if (modeMap[m]) get(modeMap[m]).classList.add('mode-on');
    toast(`${['', 'Queue', 'Emotion AI', 'Shuffle'][m]} mode activated`, 'info');
    if (m === 2) getTime();
    if (m === 3) rand_play();
}

// ┌──────────────────────────────────────────────┐
// │  FILTERS & SORTING                            │
// └──────────────────────────────────────────────┘
function filterByMood(val) {
    qsa('.chip').forEach(c => c.classList.remove('active'));
    let count = 0;
    const sel = APP.viewMode === 'grid' ? '.song-card' : '.sl-item';
    qsa(sel).forEach(card => {
        const moods = card.dataset.moods || '';
        let show = false;
        if (val === 'all') show = true;
        else if (val === 'favorites') {
            const idx = parseInt(card.id.split('-')[1]);
            show = APP.favorites.has(idx);
        }
        else show = moods.includes(val);

        if (show) { card.classList.remove('hidden'); card.style.display = ''; count++; }
        else { card.classList.add('hidden'); card.style.display = 'none'; }
    });
    get('emptyBox').style.display = count ? 'none' : '';
    updCount(count);
}

function chipFilter(el, val) {
    qsa('.chip').forEach(c => c.classList.remove('active'));
    el.classList.add('active');
    filterByMood(val);
}

function filterSongs(q) {
    q = q.toLowerCase().trim();
    let count = 0;
    const sel = APP.viewMode === 'grid' ? '.song-card' : '.sl-item';
    qsa(sel).forEach(card => {
        const t = card.dataset.title || '';
        const m = card.dataset.moods || '';
        if (!q || t.includes(q) || m.includes(q)) {
            card.classList.remove('hidden'); card.style.display = ''; count++;
        } else {
            card.classList.add('hidden'); card.style.display = 'none';
        }
    });
    get('emptyBox').style.display = count ? 'none' : '';
    updCount(count);
}

function sortSongs(mode) {
    APP.sortMode = mode;
    qsa('.sort-opt').forEach(o => o.classList.remove('active'));
    const menu = get('sortMenu');
    if (menu) { menu.classList.remove('open'); }
    const label = get('sortLabel');
    if (label) label.textContent = { default: 'Default', title: 'A-Z', 'title-desc': 'Z-A', mood: 'Mood' }[mode] || 'Default';

    if (mode === 'default') {
        renderGrid(); renderList(); return;
    }

    const container = APP.viewMode === 'grid' ? get('songsGrid') : get('songsList');
    const items = [...container.children];

    items.sort((a, b) => {
        const tA = a.dataset.title || '';
        const tB = b.dataset.title || '';
        const mA = a.dataset.moods || '';
        const mB = b.dataset.moods || '';
        if (mode === 'title') return tA.localeCompare(tB);
        if (mode === 'title-desc') return tB.localeCompare(tA);
        if (mode === 'mood') return mA.localeCompare(mB);
        return 0;
    });

    items.forEach(el => container.appendChild(el));
}

function toggleSortMenu() {
    const menu = get('sortMenu');
    if (menu) menu.classList.toggle('open');
}

function updCount(c) {
    const n = c !== undefined ? c : APP.songs.length;
    get('songCount').textContent = `${n} song${n !== 1 ? 's' : ''}`;
}

// ┌──────────────────────────────────────────────┐
// │  EMOTION / RANDOM                             │
// └──────────────────────────────────────────────┘
eel.expose(updateFrame);
function updateFrame(data) {
    const now = performance.now();
    if (now - APP._lastFrameAt < APP._frameThrottleMs) return;
    APP._lastFrameAt = now;
    const img = get('camFrame');
    if (img) img.src = data;
}

function _startEmotionStatusPoll() {
    if (APP._emotionPollTimer) return;
    const moodEl = get('detectedMood');
    const confEl = get('emotionConfidence');
    const emojiEl = get('emotionEmoji');
    const ringCircle = get('emotionRingCircle');

    APP._emotionPollTimer = setInterval(async () => {
        try {
            const s = await eel.getCameraStatus()();
            if (!s) return;
            const emo = (s.emotion || 'neutral');
            const conf = Math.max(0, Math.min(1, Number(s.confidence || 0)));
            if (moodEl) moodEl.textContent = emo.toUpperCase();
            if (confEl) confEl.textContent = `${Math.round(conf * 100)}% confidence`;
            if (emojiEl) emojiEl.textContent = emojiFor(emo);
            if (ringCircle) {
                const circumference = 2 * Math.PI * 54;
                ringCircle.style.strokeDashoffset = circumference * (1 - conf);
            }
            setLastDetectedMood(emo, conf);
        } catch (e) { }
    }, 250);
}

function _stopEmotionStatusPoll() {
    if (APP._emotionPollTimer) {
        clearInterval(APP._emotionPollTimer);
        APP._emotionPollTimer = null;
    }
}

function openEmotionModal() {
    get('emotionModal').classList.add('open');
    get('detectedMood').textContent = 'Analyzing...';
    const confEl = get('emotionConfidence');
    if (confEl) confEl.textContent = 'Please look at the camera';
    eel.startCamera()();
    _startEmotionStatusPoll();
}

function closeEmotionModal() {
    get('emotionModal').classList.remove('open');
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
    if (!APP.mood[v].length) { rand_play(); return; }
    playSong(APP.mood[v][Math.floor(Math.random() * APP.mood[v].length)]);
}

function rand_play() {
    if (!APP.songs.length) return;
    playSong(Math.floor(Math.random() * APP.songs.length));
}

// ┌──────────────────────────────────────────────┐
// │  AUDIO ENGINE                                 │
// └──────────────────────────────────────────────┘
function initAudio() {
    const a = get('main_slider');
    const savedVol = localStorage.getItem('moodify-volume');
    a.volume = savedVol !== null ? savedVol / 100 : 0.8;
    const r = get('volRange');
    if (r) r.value = a.volume * 100;

    a.addEventListener('timeupdate', () => {
        if (!a.duration) return;
        const p = (a.currentTime / a.duration) * 100;
        get('pbFill').style.width = p + '%';
        get('timeFill').style.width = p + '%';
        get('curTime').textContent = fmt(a.currentTime);
        get('totTime').textContent = fmt(a.duration);

        // Fullscreen player
        const fsFill = get('fsTimeFill');
        const fsCur = get('fsCurTime');
        const fsTot = get('fsTotTime');
        if (fsFill) fsFill.style.width = p + '%';
        if (fsCur) fsCur.textContent = fmt(a.currentTime);
        if (fsTot) fsTot.textContent = fmt(a.duration);
    });

    a.addEventListener('ended', () => {
        setPlay(false);
        if (APP.repeatMode === 2) {
            a.currentTime = 0; a.play(); setPlay(true); return;
        }
        if (APP.mod === 1) {
            if (APP.queueItems.length) nextsong();
            else if (APP.repeatMode === 1 && APP.songs.length) playSong(0);
            else toast('Queue finished', 'info');
        }
        else if (APP.mod === 2) getTime();
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
    const a = get('main_slider');
    if (a.duration) a.currentTime = pct * a.duration;
}

function setVolume(v) {
    get('main_slider').volume = v / 100;
    const ic = get('volIcon');
    ic.className = v == 0 ? 'fas fa-volume-xmark' : v < 50 ? 'fas fa-volume-low' : 'fas fa-volume-high';
    get('volRange').value = v;
    localStorage.setItem('moodify-volume', v);
}

function toggleMute() {
    const a = get('main_slider');
    const r = get('volRange');
    if (a.volume > 0) { a._pv = a.volume; a.volume = 0; r.value = 0; }
    else { a.volume = a._pv || 0.8; r.value = a.volume * 100; }
    setVolume(r.value);
}

// ┌──────────────────────────────────────────────┐
// │  AUDIO VISUALIZER (Web Audio API)             │
// └──────────────────────────────────────────────┘
let audioCtx, analyser, dataArray, sourceNode, vizConnected = false;

function initVisualizer() {
    // Canvas refs will be set up, context created on first play
}

function connectVisualizer() {
    if (vizConnected) return;
    try {
        const audio = get('main_slider');
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        analyser = audioCtx.createAnalyser();
        analyser.fftSize = 256;
        sourceNode = audioCtx.createMediaElementSource(audio);
        sourceNode.connect(analyser);
        analyser.connect(audioCtx.destination);
        dataArray = new Uint8Array(analyser.frequencyBinCount);
        vizConnected = true;
        drawVisualizer();
        drawFSVisualizer();
        console.log('[Moodify] 🎚️ Visualizer connected');
    } catch (e) {
        console.warn('[Moodify] Visualizer error:', e);
    }
}

function drawVisualizer() {
    const canvas = get('playerVisualizer');
    if (!canvas || !analyser) return;
    const ctx = canvas.getContext('2d');

    function draw() {
        requestAnimationFrame(draw);
        if (!analyser) return;
        canvas.width = canvas.offsetWidth;
        canvas.height = canvas.offsetHeight;
        analyser.getByteFrequencyData(dataArray);
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        const barCount = 64;
        const barWidth = canvas.width / barCount;
        const step = Math.floor(dataArray.length / barCount);

        for (let i = 0; i < barCount; i++) {
            const val = dataArray[i * step] / 255;
            const h = val * canvas.height * 0.6;
            const x = i * barWidth;

            const gradient = ctx.createLinearGradient(0, canvas.height, 0, canvas.height - h);
            gradient.addColorStop(0, 'rgba(29, 185, 84, 0.1)');
            gradient.addColorStop(0.5, 'rgba(102, 126, 234, 0.15)');
            gradient.addColorStop(1, 'rgba(240, 147, 251, 0.2)');

            ctx.fillStyle = gradient;
            ctx.fillRect(x, canvas.height - h, barWidth - 1, h);
        }
    }
    draw();
}

function drawFSVisualizer() {
    const canvas = get('fsVisualizer');
    if (!canvas || !analyser) return;
    const ctx = canvas.getContext('2d');

    function draw() {
        requestAnimationFrame(draw);
        if (!analyser) return;
        canvas.width = canvas.offsetWidth * 2;
        canvas.height = canvas.offsetHeight * 2;
        ctx.scale(2, 2);
        analyser.getByteFrequencyData(dataArray);
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        const w = canvas.width / 2;
        const h = canvas.height / 2;
        const cx = w / 2;
        const cy = h / 2;
        const radius = Math.min(cx, cy) * 0.7;
        const bars = 60;

        for (let i = 0; i < bars; i++) {
            const step = Math.floor(dataArray.length / bars);
            const val = dataArray[i * step] / 255;
            const angle = (i / bars) * Math.PI * 2 - Math.PI / 2;
            const barH = val * radius * 0.5;

            const x1 = cx + Math.cos(angle) * radius;
            const y1 = cy + Math.sin(angle) * radius;
            const x2 = cx + Math.cos(angle) * (radius + barH);
            const y2 = cy + Math.sin(angle) * (radius + barH);

            ctx.beginPath();
            ctx.moveTo(x1, y1);
            ctx.lineTo(x2, y2);
            ctx.strokeStyle = `hsla(${(i / bars) * 360}, 80%, 65%, ${0.4 + val * 0.6})`;
            ctx.lineWidth = 2;
            ctx.lineCap = 'round';
            ctx.stroke();
        }
    }
    draw();
}

// ┌──────────────────────────────────────────────┐
// │  FULLSCREEN PLAYER                            │
// └──────────────────────────────────────────────┘
function openFullscreen() {
    if (APP.currentSongIndex < 0) return;
    get('fsPlayer').classList.add('open');
    document.body.style.overflow = 'hidden';
}

function closeFullscreen() {
    get('fsPlayer').classList.remove('open');
    document.body.style.overflow = '';
}

// ┌──────────────────────────────────────────────┐
// │  LIKE / FAVORITES                             │
// └──────────────────────────────────────────────┘
function toggleLike() {
    if (APP.currentSongIndex < 0) return;
    if (APP.favorites.has(APP.currentSongIndex)) {
        APP.favorites.delete(APP.currentSongIndex);
        toast('Removed from favorites', 'info');
    } else {
        APP.favorites.add(APP.currentSongIndex);
        toast('Added to favorites ❤️', 'success');
    }
    syncAliases();
    updateLikeUI();
    updateFavCounter();

    // Update card heart
    const cardFav = qs(`#sc-${APP.currentSongIndex} .sc-fav-btn`);
    if (cardFav) {
        const isFav = APP.favorites.has(APP.currentSongIndex);
        cardFav.classList.toggle('active', isFav);
        cardFav.innerHTML = `<i class="${isFav ? 'fas' : 'far'} fa-heart"></i>`;
    }
    savePreferences();
}

function toggleFavAt(i) {
    if (APP.favorites.has(i)) APP.favorites.delete(i);
    else APP.favorites.add(i);
    syncAliases();

    const cardFav = qs(`#sc-${i} .sc-fav-btn`);
    if (cardFav) {
        const isFav = APP.favorites.has(i);
        cardFav.classList.toggle('active', isFav);
        cardFav.innerHTML = `<i class="${isFav ? 'fas' : 'far'} fa-heart"></i>`;
    }
    if (i === APP.currentSongIndex) updateLikeUI();
    updateFavCounter();
    savePreferences();
}

// ┌──────────────────────────────────────────────┐
// │  VIEW                                         │
// └──────────────────────────────────────────────┘
function showView(v, el) {
    qsa('.sidebar-nav .nav-link').forEach(n => n.classList.remove('active'));
    if (el) el.classList.add('active');
    get('heroSection').style.display = v === 'home' ? '' : 'none';
    const recent = get('recentSection');
    if (recent) recent.style.display = (v === 'home' && APP.recentlyPlayed.length) ? '' : 'none';
}

// ┌──────────────────────────────────────────────┐
// │  UPLOAD                                       │
// └──────────────────────────────────────────────┘
function openUploadModal() { get('uploadModal').classList.add('open'); }
function closeUploadModal() {
    get('uploadModal').classList.remove('open');
    get('uploadFile').value = '';
    get('uploadTitle').value = '';
    get('uploadDesc').value = '';
    get('dzFilename').textContent = '';
    get('uploadLoading').style.display = 'none';
}

function onFileChosen(inp) {
    if (inp.files.length) {
        get('dzFilename').textContent = `✓ ${inp.files[0].name}`;
        if (!get('uploadTitle').value)
            get('uploadTitle').value = inp.files[0].name.replace('.mp3', '');
    }
}

async function handleUpload() {
    const fi = get('uploadFile');
    if (!fi.files.length) { toast('Select a file first!', 'error'); return; }
    const file = fi.files[0];
    const reader = new FileReader();
    reader.onload = async (e) => {
        const meta = {
            title: get('uploadTitle').value || file.name.replace('.mp3', ''),
            description: get('uploadDesc').value || 'Uploaded Song',
            moods: [get('uploadMood').value],
            cover_b64: ''
        };
        try {
            get('uploadLoading').style.display = '';
            let res = await eel.upload_song(e.target.result, meta)();
            if (res.success) {
                toast('Upload successful! ✨', 'success');
                closeUploadModal();
                setTimeout(() => loadSongs(true), 2000);
            } else toast('Upload failed: ' + res.message, 'error');
        } catch (err) { console.error(err); toast('Upload error', 'error'); }
        finally { get('uploadLoading').style.display = 'none'; }
    };
    reader.readAsDataURL(file);
}

function switchToYouTube() {
    closeUploadModal();
    setTimeout(() => openYouTubeModal(), 200);
}

// ┌──────────────────────────────────────────────┐
// │  YOUTUBE DOWNLOAD                             │
// └──────────────────────────────────────────────┘
function openYouTubeModal() { get('ytModal').classList.add('open'); resetYtStep(); }
function closeYouTubeModal() { get('ytModal').classList.remove('open'); resetYtStep(); }

function resetYtStep() {
    resetYtUI();
    const urlInput = get('ytUrl');
    if (urlInput) { urlInput.value = ''; urlInput.style.borderColor = ''; }
}

function resetYtUI() {
    const body = qs('#ytModal .modal-body');
    if (body) body.style.display = 'block';
    const loadingEl = get('ytLoading');
    if (loadingEl) loadingEl.style.display = 'none';
    const pBar = get('ytProgressBar');
    if (pBar) pBar.style.width = '0%';
    const pText = get('ytProgressText');
    if (pText) pText.textContent = '0%';
    const pSpeed = get('ytSpeed');
    if (pSpeed) pSpeed.textContent = '';
    get('ytStep1').style.display = 'block';
    get('ytPreview').style.display = 'none';
    get('ytMoodSection').style.display = 'none';
    get('ytDownloadBtn').style.display = 'none';
}

eel.expose(updateDownloadProgress);
function updateDownloadProgress(data) {
    const loadingEl = get('ytLoading');
    if (loadingEl && loadingEl.style.display !== 'none') {
        const pBar = get('ytProgressBar');
        if (pBar) pBar.style.width = data.percent + '%';
        const pText = get('ytProgressText');
        if (pText) pText.textContent = Math.round(data.percent) + '%';
        const pSpeed = get('ytSpeed');
        if (pSpeed) {
            pSpeed.textContent = data.status === 'finished'
                ? 'Converting...'
                : `${data.speed} • ETA: ${data.eta}`;
        }
    }
}

async function checkYouTubeUrl() {
    const url = get('ytUrl').value.trim();
    if (!url) return;
    const btn = qs('.btn-check');
    const orig = btn.innerHTML;
    try {
        btn.innerHTML = '<div class="spinner" style="width:16px;height:16px;border-width:2px"></div>';
        let res = await eel.get_youtube_info(url)();
        if (res.success) {
            const info = res.data;
            get('ytThumb').src = info.thumbnail;
            get('ytTitle').textContent = info.title;
            get('ytAuthor').textContent = info.author;
            get('ytPreview').style.display = 'flex';
            get('ytMoodSection').style.display = 'block';
            get('ytDownloadBtn').style.display = 'block';
        } else {
            toast('Could not find video', 'error');
            get('ytUrl').style.borderColor = 'red';
        }
    } catch (e) { console.error(e); toast('Connection error', 'error'); }
    finally { btn.innerHTML = orig; }
}

async function submitYouTube() {
    const url = get('ytUrl').value.trim();
    const mood = get('ytMood').value;
    if (!url) { toast('Please enter a valid URL', 'error'); return; }
    try {
        qs('#ytModal .modal-body').style.display = 'none';
        get('ytLoading').style.display = 'flex';
        get('ytLoadingText').textContent = 'Downloading & Converting...';
        let res = await eel.download_youtube_song(url, mood)();
        if (res.success) {
            toast('Download complete! 🎉', 'success');
            closeYouTubeModal();
            setTimeout(() => loadSongs(true), 1500);
        } else {
            toast('Download failed: ' + res.message, 'error');
            qs('#ytModal .modal-body').style.display = 'block';
            get('ytLoading').style.display = 'none';
        }
    } catch (e) {
        console.error(e);
        toast('Error interacting with backend', 'error');
        qs('#ytModal .modal-body').style.display = 'block';
        get('ytLoading').style.display = 'none';
    }
}

// ┌──────────────────────────────────────────────┐
// │  DRAG & DROP                                  │
// └──────────────────────────────────────────────┘
function initDragDrop() {
    const dz = get('dropZone');
    if (!dz) return;
    ['dragenter', 'dragover'].forEach(ev =>
        dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.add('dragover'); }));
    ['dragleave', 'drop'].forEach(ev =>
        dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.remove('dragover'); }));
    dz.addEventListener('drop', e => {
        if (e.dataTransfer.files.length) {
            get('uploadFile').files = e.dataTransfer.files;
            onFileChosen(get('uploadFile'));
        }
    });
}

// ┌──────────────────────────────────────────────┐
// │  KEYBOARD SHORTCUTS (Enhanced)                │
// └──────────────────────────────────────────────┘
function initKeys() {
    document.addEventListener('keydown', e => {
        if (['INPUT', 'SELECT', 'TEXTAREA'].includes(e.target.tagName)) {
            if (e.code === 'Escape') e.target.blur();
            return;
        }
        switch (e.code) {
            case 'Space': e.preventDefault(); togglePlay(); break;
            case 'ArrowRight':
                if (e.shiftKey) { // seek forward 10s
                    const a = get('main_slider');
                    if (a.duration) a.currentTime = Math.min(a.duration, a.currentTime + 10);
                } else nextsong();
                break;
            case 'ArrowLeft':
                if (e.shiftKey) { // seek back 10s
                    const a = get('main_slider');
                    a.currentTime = Math.max(0, a.currentTime - 10);
                } else prevSong();
                break;
            case 'ArrowUp': e.preventDefault();
                setVolume(Math.min(100, parseInt(get('volRange').value) + 5)); break;
            case 'ArrowDown': e.preventDefault();
                setVolume(Math.max(0, parseInt(get('volRange').value) - 5)); break;
            case 'KeyQ': toggleQueue(); break;
            case 'KeyF': e.preventDefault();
                get('fsPlayer').classList.contains('open') ? closeFullscreen() : openFullscreen(); break;
            case 'KeyM': toggleMute(); break;
            case 'KeyD': toggleTheme(); break;
            case 'KeyS': rand_play(); break;
            case 'KeyR': toggleRepeat(); break;
            case 'Escape': closeAllModals(); break;
            case 'Slash':
                if (e.shiftKey) showKeyboardShortcuts(); break;
        }
        if ((e.metaKey || e.ctrlKey) && e.code === 'KeyK') {
            e.preventDefault();
            get('searchInput').focus();
        }
    });
}

function closeAllModals() {
    qsa('.modal-dim.open').forEach(m => m.classList.remove('open'));
    closeFullscreen();
    const sidebar = get('sidebar');
    if (sidebar) sidebar.classList.remove('mobile-open');
    const qd = get('queueDrawer');
    if (qd && qd.classList.contains('open')) toggleQueue();
}

function showKeyboardShortcuts() {
    get('shortcutsModal').classList.add('open');
}

// ┌──────────────────────────────────────────────┐
// │  SCROLL ANIMATIONS (Intersection Observer)    │
// └──────────────────────────────────────────────┘
function initScrollAnimations() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.1, rootMargin: '0px 0px -50px 0px' });

    qsa('.fade-in-up, .section-title, .hero-left, .hero-right').forEach(el => {
        observer.observe(el);
    });
}

// ┌──────────────────────────────────────────────┐
// │  TOPBAR SCROLL EFFECT                         │
// └──────────────────────────────────────────────┘
function initTopbarScroll() {
    const topbar = get('topbar');
    const main = get('mainArea');
    if (!main || !topbar) return;
    main.addEventListener('scroll', () => {
        topbar.classList.toggle('scrolled', main.scrollTop > 20);
    });
    window.addEventListener('scroll', () => {
        topbar.classList.toggle('scrolled', window.scrollY > 20);
    });
}

// ┌──────────────────────────────────────────────┐
// │  ANIMATED COUNTERS                            │
// └──────────────────────────────────────────────┘
function animateCounters() {
    qsa('[data-animate-counter]').forEach(stat => {
        const numEl = stat.querySelector('.stat-num');
        if (!numEl) return;
        const target = parseInt(numEl.getAttribute('data-target') || numEl.textContent);
        if (isNaN(target)) return;
        animateNumber(numEl, 0, target, 1200);
    });
}

function animateNumber(el, start, end, duration) {
    const startTime = performance.now();
    function update(now) {
        const elapsed = now - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3); // easeOutCubic
        const current = Math.round(start + (end - start) * eased);
        el.textContent = current;
        if (progress < 1) requestAnimationFrame(update);
    }
    requestAnimationFrame(update);
}

// ┌──────────────────────────────────────────────┐
// │  PARTICLE SYSTEM (Hero Background)            │
// └──────────────────────────────────────────────┘
function initParticles() {
    const canvas = get('heroParticles');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let particles = [];
    const colors = ['#1DB954', '#667eea', '#f093fb', '#fbbf24', '#06b6d4'];

    function resize() {
        const hero = get('heroSection');
        if (!hero) return;
        canvas.width = hero.offsetWidth;
        canvas.height = hero.offsetHeight;
    }

    function createParticle() {
        return {
            x: Math.random() * canvas.width,
            y: Math.random() * canvas.height,
            r: Math.random() * 3 + 1,
            dx: (Math.random() - 0.5) * 0.5,
            dy: (Math.random() - 0.5) * 0.5,
            color: colors[Math.floor(Math.random() * colors.length)],
            alpha: Math.random() * 0.5 + 0.1,
        };
    }

    function init() {
        resize();
        particles = [];
        for (let i = 0; i < 40; i++) particles.push(createParticle());
    }

    function draw() {
        requestAnimationFrame(draw);
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        particles.forEach(p => {
            p.x += p.dx;
            p.y += p.dy;
            if (p.x < 0 || p.x > canvas.width) p.dx *= -1;
            if (p.y < 0 || p.y > canvas.height) p.dy *= -1;

            ctx.beginPath();
            ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
            ctx.fillStyle = p.color;
            ctx.globalAlpha = p.alpha;
            ctx.fill();
            ctx.globalAlpha = 1;
        });

        // Draw connections
        for (let i = 0; i < particles.length; i++) {
            for (let j = i + 1; j < particles.length; j++) {
                const dx = particles[i].x - particles[j].x;
                const dy = particles[i].y - particles[j].y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < 120) {
                    ctx.beginPath();
                    ctx.moveTo(particles[i].x, particles[i].y);
                    ctx.lineTo(particles[j].x, particles[j].y);
                    ctx.strokeStyle = particles[i].color;
                    ctx.globalAlpha = 0.05 * (1 - dist / 120);
                    ctx.lineWidth = 0.5;
                    ctx.stroke();
                    ctx.globalAlpha = 1;
                }
            }
        }
    }

    init();
    draw();
    window.addEventListener('resize', resize);
}

// ┌──────────────────────────────────────────────┐
// │  SETTINGS MENU                                │
// └──────────────────────────────────────────────┘
function toggleSettingsMenu() {
    const dd = get('settingsDropdown');
    if (dd) dd.classList.toggle('open');
}

// Close settings dropdown when clicking outside
document.addEventListener('click', (e) => {
    const dd = get('settingsDropdown');
    const pill = qs('.user-pill');
    if (dd && dd.classList.contains('open') && !pill.contains(e.target)) {
        dd.classList.remove('open');
    }
    const sortMenu = get('sortMenu');
    const sortWrap = qs('.sort-dropdown-wrap');
    if (sortMenu && sortMenu.classList.contains('open') && sortWrap && !sortWrap.contains(e.target)) {
        sortMenu.classList.remove('open');
    }
});

// ┌──────────────────────────────────────────────┐
// │  TOAST SYSTEM                                 │
// └──────────────────────────────────────────────┘
function toast(msg, type = 'info') {
    const c = get('toastStack');
    if (!c) return;
    const t = document.createElement('div');
    t.className = `toast-item ${type}`;
    const icons = {
        success: 'fa-circle-check',
        error: 'fa-circle-xmark',
        warning: 'fa-triangle-exclamation',
        info: 'fa-circle-info'
    };
    t.innerHTML = `
        <i class="fas ${icons[type] || icons.info}"></i>
        <span>${msg}</span>
        <div class="toast-progress"><div class="toast-progress-fill ${type}"></div></div>
    `;
    t.onclick = () => t.remove();
    c.appendChild(t);
    setTimeout(() => { if (t.parentNode) t.remove(); }, 4000);
}

// ┌──────────────────────────────────────────────┐
// │  SYSTEM                                       │
// └──────────────────────────────────────────────┘
eel.expose(close_app);
function close_app() { window.close(); }

window.onbeforeunload = function () { savePreferences(); };

// Mood pill colors
const moodPillStyles = document.createElement('style');
moodPillStyles.textContent = `
    .mood-pill.happy { background: rgba(251,191,36,0.15); }
    .mood-pill.happy .mp-dot { background: #fbbf24; }
    .mood-pill.sad { background: rgba(59,130,246,0.15); }
    .mood-pill.sad .mp-dot { background: #3b82f6; }
    .mood-pill.angry { background: rgba(239,68,68,0.15); }
    .mood-pill.angry .mp-dot { background: #ef4444; }
    .mood-pill.neutral { background: rgba(16,185,129,0.15); }
    .mood-pill.neutral .mp-dot { background: #10b981; }
    .mood-pill.surprise { background: rgba(139,92,246,0.15); }
    .mood-pill.surprise .mp-dot { background: #8b5cf6; }
`;
document.head.appendChild(moodPillStyles);