document.addEventListener('DOMContentLoaded', () => {
    const predictBtn = document.getElementById('predict-btn');
    const toggleAdvanced = document.getElementById('toggle-advanced');
    const advancedFeatures = document.getElementById('advanced-features');
    const loading = document.getElementById('loading');
    const results = document.getElementById('results');
    const resultContent = document.getElementById('result-content');

    toggleAdvanced.addEventListener('click', () => {
        advancedFeatures.classList.toggle('visible');
        toggleAdvanced.textContent = advancedFeatures.classList.contains('visible')
            ? 'Ocultar Features Avanzadas'
            : 'Mostrar/Ocultar Features Avanzadas';
    });

    predictBtn.addEventListener('click', async () => {
        const apiKey = document.getElementById('api-key').value.trim();
        const homeTeam = document.getElementById('home-team').value.trim();
        const awayTeam = document.getElementById('away-team').value.trim();
        const league = document.getElementById('league').value;

        if (!homeTeam && league === 'random') {
            alert('Introduce al menos los nombres de los equipos, o selecciona una liga para datos aleatorios.');
            return;
        }

        const features = buildFeatures(homeTeam, awayTeam, league);
        loading.classList.remove('hidden');
        results.classList.add('hidden');
        predictBtn.disabled = true;

        try {
            let fixtures = null;
            if (apiKey && homeTeam && awayTeam) {
                fixtures = await fetchFixtures(apiKey, homeTeam, awayTeam, league);
            }

            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    home_team: homeTeam || undefined,
                    away_team: awayTeam || undefined,
                    league: league === 'random' ? 'laliga' : league,
                    features: features,
                    use_mock: !homeTeam || !awayTeam
                })
            });

            if (!response.ok) throw new Error(`Server error: ${response.status}`);
            const data = await response.json();
            displayResults(data, fixtures);

        } catch (error) {
            console.error('Prediction error:', error);
            const mockData = generateMockPredictions(homeTeam || 'Local', awayTeam || 'Visitante', league);
            displayResults(mockData, null, true);
        } finally {
            loading.classList.add('hidden');
            predictBtn.disabled = false;
        }
    });

    function buildFeatures(homeTeam, awayTeam, league) {
        const xgHome = parseFloat(document.getElementById('xg-home').value) || null;
        const xgAway = parseFloat(document.getElementById('xg-away').value) || null;
        const eloHome = parseInt(document.getElementById('elo-home').value) || null;
        const eloAway = parseInt(document.getElementById('elo-away').value) || null;
        return {
            xg_home: xgHome, xg_away: xgAway,
            elo_home: eloHome, elo_away: eloAway,
            home_advantage: 0.15, is_derby: false
        };
    }

    async function fetchFixtures(apiKey, homeTeam, awayTeam, league) {
        const response = await fetch(`https://api.api-football.com/v3/fixtures?team=${homeTeam}`, {
            headers: { 'x-rapidapi-host': 'api-football.com', 'x-rapidapi-key': apiKey }
        });
        if (!response.ok) return null;
        return response.json();
    }

    function displayResults(data, fixtures, isMock = false) {
        resultContent.innerHTML = `
            <div class="result-prob">
                <div class="result-prob-item home">
                    <div class="label">${data.home_team || 'Local'} (Gana)</div>
                    <div class="value">${(data.prob_home * 100).toFixed(1)}%</div>
                </div>
                <div class="result-prob-item draw">
                    <div class="label">Empate</div>
                    <div class="value">${(data.prob_draw * 100).toFixed(1)}%</div>
                </div>
                <div class="result-prob-item away">
                    <div class="label">${data.away_team || 'Visitante'} (Gana)</div>
                    <div class="value">${(data.prob_away * 100).toFixed(1)}%</div>
                </div>
            </div>
            <div class="result-markets">
                <div class="market-result">
                    <span class="market-name">BTTS (Ambos marcan)</span>
                    <span class="market-value">${(data.prob_btts * 100).toFixed(1)}%</span>
                </div>
                <div class="market-result">
                    <span class="market-name">Over 2.5</span>
                    <span class="market-value">${(data.ou['2'] * 100).toFixed(1)}%</span>
                </div>
                <div class="market-result">
                    <span class="market-name">Under 2.5</span>
                    <span class="market-value">${((1 - data.ou['2']) * 100).toFixed(1)}%</span>
                </div>
                <div class="market-result">
                    <span class="market-name">xG Local</span>
                    <span class="market-value">${data.xg_home.toFixed(2)}</span>
                </div>
                <div class="market-result">
                    <span class="market-name">xG Visitante</span>
                    <span class="market-value">${data.xg_away.toFixed(2)}</span>
                </div>
                <div class="market-result">
                    <span class="market-name">Confianza</span>
                    <span class="market-value">${data.confidence}</span>
                </div>
            </div>
            ${data.top_scores ? `
            <h4 style="margin: 20px 0 12px;">Resultados Exactos Mas Probables</h4>
            <div class="result-markets">
                ${data.top_scores.map(s => `
                    <div class="market-result">
                        <span class="market-name">${s.score}</span>
                        <span class="market-value">${s.prob_pct.toFixed(1)}%</span>
                    </div>
                `).join('')}
            </div>` : ''}
            ${isMock ? '<div style="margin-top:16px;padding:12px;background:#fef3c7;border-radius:8px;font-size:0.9rem;color:#92400e;text-align:center;">Resultados simulados (demo sin servidor ONNX). Para la demo interactiva, el servidor ONNX debe estar en funcionamiento.</div>' : ''}
        `;
        results.classList.remove('hidden');
    }

    function generateMockPredictions(home, away, league) {
        const baseHome = 0.45, baseDraw = 0.28, baseAway = 0.27;
        const noise = () => (Math.random() - 0.5) * 0.15;
        const prob_home = Math.max(0.1, Math.min(0.7, baseHome + noise()));
        const prob_draw = Math.max(0.15, Math.min(0.35, baseDraw + noise()));
        const prob_away = 1 - prob_home - prob_draw;
        const xg_home = (1.2 + Math.random() * 1.0).toFixed(2);
        const xg_away = (0.8 + Math.random() * 0.8).toFixed(2);
        const prob_btts = 0.45 + Math.random() * 0.2;
        const prob_over25 = 0.50 + Math.random() * 0.2;
        const confidence = ['ALTA', 'MEDIA', 'BAJA'][Math.floor(Math.random() * 3)];
        const scores = ['1-0', '0-0', '1-1', '2-1', '0-1', '2-0', '1-2', '0-2'];
        const top_scores = scores.map(score => ({
            score, prob: 0.03 + Math.random() * 0.12, prob_pct: 0
        }));
        const total = top_scores.reduce((s, sc) => s + sc.prob, 0);
        top_scores.forEach(sc => sc.prob_pct = (sc.prob / total * 100));
        top_scores.sort((a, b) => b.prob - a.prob);
        return {
            home_team: home, away_team: away,
            prob_home: Math.round(prob_home * 10000) / 10000,
            prob_draw: Math.round(prob_draw * 10000) / 10000,
            prob_away: Math.round(prob_away * 10000) / 10000,
            prob_btts: Math.round(prob_btts * 10000) / 10000,
            ou: { '0': 0.95, '1': 0.75, '2': prob_over25, '3': 0.45, '4': 0.25 },
            xg_home: parseFloat(xg_home), xg_away: parseFloat(xg_away),
            confidence, top_scores: top_scores.slice(0, 6)
        };
    }
});
