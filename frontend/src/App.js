import React, { useState } from 'react';
import axios from 'axios';

// --- Simple SVG chart components (no external libs) ---
function SimpleBarChart({ labels = [], values = [], colors = [] , height=180}){
  const max = Math.max(...values, 1);
  const w = 380;
  const barW = Math.max(20, Math.floor(w / Math.max(1, values.length)) - 8);
  return (
    <div className="chart">
      <svg width="100%" height={height} viewBox={`0 0 ${w} ${height}`} className="chart-svg">
        {values.map((v,i)=>{
          const x = 12 + i * (barW + 8);
          const h = Math.round((v / max) * (height - 40));
          const y = height - h - 24;
          const fill = colors[i] || `hsl(${(i*55)%360} 80% 60%)`;
          return (
            <g key={i}>
              <rect x={x} y={y} width={barW} height={h} rx={6} fill={fill} />
              <text x={x + barW/2} y={height-8} textAnchor="middle" fontSize={10} fill="#333">{labels[i]}</text>
            </g>
          )
        })}
      </svg>
    </div>
  )
}

function SimpleLineChart({labels=[], values=[], stroke='rgba(255,122,45,0.95)', height=180}){
  const max = Math.max(...values, 1);
  const w = 420;
  const step = Math.max(1, w / Math.max(1, values.length-1));
  const points = values.map((v,i)=>`${12 + i*step},${(height-24) - (v/max)*(height-44)}`).join(' ');
  return (
    <div className="chart">
      <svg width="100%" height={height} viewBox={`0 0 ${w} ${height}`} className="chart-svg">
        <polyline points={points} fill="none" stroke={stroke} strokeWidth={3} strokeLinecap="round" strokeLinejoin="round" />
        {values.map((v,i)=>{
          const x = 12 + i*step;
          const y = (height-24) - (v/max)*(height-44);
          return <circle key={i} cx={x} cy={y} r={4} fill={stroke} />;
        })}
        {labels.map((l,i)=> <text key={i} x={12 + i*step} y={height-6} textAnchor="middle" fontSize={10} fill="#333">{l}</text>)}
      </svg>
    </div>
  )
}

export default function App() {
  const [started, setStarted] = useState(false);
  const [product, setProduct] = useState('Smartwatch');
  const [brand, setBrand] = useState('Sony');
  const [category, setCategory] = useState('Electronics');
  const [futureStart, setFutureStart] = useState('2025-01-01');
  const [futureEnd, setFutureEnd] = useState('2025-01-31');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [tab, setTab] = useState('forecast');
  const [error, setError] = useState(null);

  const submit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      let analyzeData = null;
      if (futureStart && futureEnd) {
        try {
          const aresp = await axios.post('/analyze', { future_start: futureStart, future_end: futureEnd });
          analyzeData = aresp.data;
        } catch (ae) {
          analyzeData = { error: ae?.response?.data || ae?.message };
        }
      }

      const cresp = await axios.post('/campaign', { product_name: product, brand, category });
      const cdata = cresp.data || {};

      // Normalize campaign output
      let outputObj = null;
      if (cdata.output) outputObj = cdata.output;
      else if (cdata.predicted_segment || cdata.recommended_platforms_ranked) outputObj = cdata;

      // Normalize forecast: support both old flat keys and nested `forecast` object
      let forecastObj = null;
      if (analyzeData) {
        if (analyzeData.forecast) {
          forecastObj = analyzeData.forecast;
        } else if (analyzeData.future_total_cost_prediction || analyzeData.historical_top_products) {
          // build forecast object from top-level fields
          forecastObj = {
            future_total_cost_prediction: analyzeData.future_total_cost_prediction,
            historical_top_products: analyzeData.historical_top_products || [],
            predicted_top_products: analyzeData.predicted_top_products || [],
            ai_generated_advice: analyzeData.ai_generated_advice || null,
            important_advice: analyzeData.important_advice || null
          };
        } else if (analyzeData.error) {
          forecastObj = { error: analyzeData.error };
        }
      }

      // csv/json names returned as basenames; build download URLs
      const csvName = cdata.csv_path || cdata.csvPath || null;
      const jsonName = cdata.json_path || cdata.jsonPath || null;
      const csvUrl = csvName ? `/reports/${csvName}` : null;
      const jsonUrl = jsonName ? `/reports/${jsonName}` : null;

      setResult({ output: outputObj, csv_path: csvName, json_path: jsonName, csv_url: csvUrl, json_url: jsonUrl, forecast: forecastObj });
    } catch (err) {
      setError(err.message || 'Request failed');
    } finally {
      setLoading(false);
    }
  };

  if (!started) {
    return (
      <div className="welcome-page">
        <div className="welcome-card">
          <div className="welcome-left">
            <h1>Welcome</h1>
            <h2>Walmart AI Agent</h2>
            <p className="subtitle">Transform sales and marketing with AI-driven forecasts, customer segmentation, and campaign planning. Get tailored platform recommendations, cost estimates, and downloadable campaign reports.</p>
            <button className="primary get-started" onClick={()=>setStarted(true)}>Get Started</button>
          </div>
          <div className="welcome-right">
            <img src="/assets/ai_robo.webp" alt="AI Robo" className="welcome-robot"/>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="page">
      <aside className="left">
        <h1>Input Section</h1>
        <p className="subtitle">Enter product details</p>
        <form onSubmit={submit} className="form">
          <label>Product Name</label>
          <input value={product} onChange={e=>setProduct(e.target.value)} />
          <label>Brand</label>
          <input value={brand} onChange={e=>setBrand(e.target.value)} />
          <label>Category</label>
          <input value={category} onChange={e=>setCategory(e.target.value)} />
          <label>Forecast Start (YYYY-MM-DD)</label>
          <input value={futureStart} onChange={e=>setFutureStart(e.target.value)} />
          <label>Forecast End (YYYY-MM-DD)</label>
          <input value={futureEnd} onChange={e=>setFutureEnd(e.target.value)} />

          <button className="primary" disabled={loading}>{loading? 'Planning...' : 'Forcast & Plan Campaign'}</button>
        </form>

        {error && <div className="error">{error}</div>}
      </aside>

      <main className={`right ${tab}-tab`}>
        <div className="header">
          <div className="brand">
            <div className="logo">W</div>
            <h1>Walmart AI Agent</h1>
          </div>
          <div className="dev-tab">Login</div>
        </div>

        <div className="output">
          {!result && <div className="placeholder">Results will appear here</div>}

          {result && (
            <div>
              <div className="tabs">
                <button className={tab==='forecast' ? 'tab active' : 'tab'} onClick={()=>setTab('forecast')}>Forecast</button>
                <button className={tab==='campaign' ? 'tab active' : 'tab'} onClick={()=>setTab('campaign')}>Campaign</button>
                  <button className={tab==='analysis' ? 'tab active' : 'tab'} onClick={()=>setTab('analysis')}>Analysis</button>
              </div>

              {tab === 'forecast' && (
                <div>
                  {result.forecast && !result.forecast.error ? (
                    <>
                      <h2>Forecast Summary</h2>
                      <div className="card-grid">
                        <div className="metric">
                          <div className="muted">Future Total Cost</div>
                          <div className="value">${(result.forecast.future_total_cost_prediction || 0).toLocaleString()}</div>
                        </div>
                        <div className="list-card">
                          <div className="muted">Historical Top Products</div>
                          <div className="product-list">
                            {(result.forecast.historical_top_products || []).slice(0,5).map((p,i)=> (
                              <div className="product-item" key={i}>{p.Product_Name} — ${Number(p.Purchase_Amount).toLocaleString()}</div>
                            ))}
                          </div>
                        </div>

                        <div className="list-card">
                          <div className="muted">Predicted Top Products</div>
                          <div className="product-list">
                            {(result.forecast.predicted_top_products || []).slice(0,5).map((p,i)=> (
                              <div className="product-item" key={i}>{p.Product_Name} — ${Number(p.Predicted_Amount).toLocaleString(undefined,{maximumFractionDigits:2})}</div>
                            ))}
                          </div>
                        </div>
                      </div>

                      <div style={{marginTop:12}} className="list-card">
                        <strong>AI Insights</strong>
                        <p style={{marginTop:8}}>{result.forecast.ai_generated_advice}</p>
                        <strong>Important Advice</strong>
                        <p className="important-advice" style={{marginTop:8}}>{result.forecast.important_advice}</p>
                      </div>
                    </>
                  ) : (
                    <div className="error">Forecast error: {JSON.stringify(result.forecast?.error || 'No forecast')}</div>
                  )}
                </div>
              )}

              {tab === 'campaign' && (
                <div>
                  <div className="card-grid">
                    <div className="metric">
                      <div className="muted">Predicted Segment</div>
                      <div style={{marginTop:8}} className="value">{(result.output?.predicted_segment?.Age_Group || '')} • {result.output?.predicted_segment?.Gender || ''}</div>
                    </div>

                    <div className="list-card">
                      <div className="muted">Top Platforms</div>
                      <div style={{marginTop:8}}>
                        {(result.output?.recommended_platforms_ranked || []).slice(0,8).map((p, idx) => (
                          <div key={p} className="platform">
                            <div style={{display:'flex',gap:8,alignItems:'center'}}>
                              <div className="badge">#{idx+1}</div>
                              <div>{p}</div>
                            </div>
                            <div style={{fontWeight:700}}>${(result.output?.estimated_costs_usd?.[p] || 0).toFixed(2)}</div>
                          </div>
                        ))}
                      </div>
                    </div>

                    <div className="list-card">
                      <div className="muted">Estimated Costs</div>
                      <div style={{marginTop:8}}>
                        {Object.entries(result.output?.estimated_costs_usd || {}).slice(0,8).map(([k,v]) => (
                          <div key={k} className="platform">
                            <div>{k}</div>
                            <div style={{fontWeight:800}}>${v}</div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>

                  <div style={{marginTop:12}} className="list-card">
                    <strong>Campaign Advice</strong>
                    <ul style={{marginTop:8}}>
                      {(result.output?.advice || []).map((a, i) => <li key={i}>{a}</li>)}
                    </ul>
                    <div style={{marginTop:12}}>
                      {result.csv_url ? <a className="download" href={result.csv_url} target="_blank" rel="noreferrer">Download CSV</a> : <span style={{color:'#999'}}>CSV: N/A</span>} &nbsp; 
                      {result.json_url ? <a className="download" href={result.json_url} target="_blank" rel="noreferrer">Download JSON</a> : <span style={{color:'#999'}}>JSON: N/A</span>}
                    </div>
                  </div>
                </div>
              )}

              {tab === 'analysis' && (
                <div>
                  <h2>Analysis Visualizations</h2>
                  <div className="card-grid">
                    <div className="list-card">
                      <div className="muted">Forecast — Predicted Top Products (Bar)</div>
                      {result?.forecast?.predicted_top_products?.length ? (
                        <SimpleBarChart
                          labels={result.forecast.predicted_top_products.map(p=>p.Product_Name)}
                          values={result.forecast.predicted_top_products.map(p=>Number(p.Predicted_Amount||0))}
                        />
                      ) : <div className="placeholder">No forecast data</div>}
                    </div>

                    <div className="list-card">
                      <div className="muted">Forecast — Historical Top Products (Line)</div>
                      {result?.forecast?.historical_top_products?.length ? (
                        <SimpleLineChart
                          labels={result.forecast.historical_top_products.map(p=>p.Product_Name)}
                          values={result.forecast.historical_top_products.map(p=>Number(p.Purchase_Amount||0))}
                          stroke={'#1e90ff'}
                        />
                      ) : <div className="placeholder">No historical data</div>}
                    </div>

                    <div className="list-card">
                      <div className="muted">Campaign — Estimated Costs per Platform (Bar)</div>
                      {result?.output?.estimated_costs_usd ? (
                        <SimpleBarChart
                          labels={Object.keys(result.output.estimated_costs_usd)}
                          values={Object.values(result.output.estimated_costs_usd).map(v=>Number(v||0))}
                        />
                      ) : <div className="placeholder">No campaign cost data</div>}
                    </div>

                    <div className="list-card">
                      <div className="muted">Campaign — Cumulative Cost (Line)</div>
                      {result?.output?.estimated_costs_usd ? (
                        (()=>{
                          const vals = Object.values(result.output.estimated_costs_usd).map(v=>Number(v||0));
                          const cum = vals.map((_,i)=>vals.slice(0,i+1).reduce((a,b)=>a+b,0));
                          return <SimpleLineChart labels={Object.keys(result.output.estimated_costs_usd)} values={cum} stroke={'#ff7a2d'} />
                        })()
                      ) : <div className="placeholder">No campaign cost data</div>}
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

      </main>
    </div>
  );
}
