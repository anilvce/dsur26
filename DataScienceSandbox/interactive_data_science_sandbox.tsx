import React, { useState, useEffect } from 'react';
import { 
  Database, 
  Trash2, 
  BarChart3, 
  Brain, 
  Play, 
  CheckCircle, 
  RefreshCw, 
  Sliders, 
  HelpCircle, 
  TrendingUp, 
  Sparkles, 
  ChevronRight, 
  Info, 
  Code, 
  Server, 
  Check, 
  AlertCircle
} from 'lucide-react';

// --- DATASET DEFINITIONS & GENERATORS ---
const DATASETS = {
  housing: {
    name: "Real Estate Price Predictor",
    objective: "Predict home sale prices based on physical attributes and locations to optimize real estate portfolios.",
    metricName: "R² Score & RMSE",
    features: [
      { id: 'sqft', name: 'Square Footage', min: 500, max: 5000, default: 2200, unit: 'sq ft' },
      { id: 'bedrooms', name: 'Bedrooms', min: 1, max: 6, default: 3, unit: '' },
      { id: 'age', name: 'Property Age', min: 0, max: 80, default: 15, unit: 'years' },
      { id: 'location_score', name: 'Location Grade', min: 1, max: 10, default: 7, unit: '/10' }
    ],
    rawSample: [
      { id: 1, sqft: 2100, bedrooms: 3, age: 12, location_score: 8, price: 420000 },
      { id: 2, sqft: null, bedrooms: 4, age: 25, location_score: 5, price: 310000 },
      { id: 3, sqft: 1400, bedrooms: 2, age: 40, location_score: 6, price: 215000 },
      { id: 4, sqft: 3800, bedrooms: 5, age: 2, location_score: 9, price: 890000 },
      { id: 5, sqft: 900, bedrooms: 1, age: 50, location_score: 3, price: 110000 },
      { id: 6, sqft: 2800, bedrooms: null, age: 18, location_score: 7, price: 510000 },
      { id: 7, sqft: 1800, bedrooms: 3, age: 8, location_score: 10, price: 540000 },
      { id: 8, sqft: 4500, bedrooms: 6, age: 1, location_score: 9, price: null }
    ]
  },
  churn: {
    name: "Telecom Customer Churn Guard",
    objective: "Predict which high-value subscribers are likely to cancel services to enable proactive retention campaigns.",
    metricName: "F1-Score & Accuracy",
    features: [
      { id: 'tenure', name: 'Tenure (Months)', min: 1, max: 72, default: 24, unit: 'months' },
      { id: 'monthly_charges', name: 'Monthly Charges', min: 20, max: 120, default: 75, unit: '$' },
      { id: 'support_calls', name: 'Support Calls (Last Month)', min: 0, max: 10, default: 2, unit: 'calls' },
      { id: 'contract_type', name: 'Contract Type (1=Year, 0=Month-to-Month)', min: 0, max: 1, default: 0, unit: '' }
    ],
    rawSample: [
      { id: 1, tenure: 12, monthly_charges: 85, support_calls: 4, contract_type: 0, churn: 1 },
      { id: 2, tenure: 48, monthly_charges: 40, support_calls: 0, contract_type: 1, churn: 0 },
      { id: 3, tenure: null, monthly_charges: 95, support_calls: 3, contract_type: 0, churn: 1 },
      { id: 4, tenure: 2, monthly_charges: 110, support_calls: 6, contract_type: 0, churn: 1 },
      { id: 5, tenure: 60, monthly_charges: 65, support_calls: 1, contract_type: 1, churn: 0 },
      { id: 6, tenure: 24, monthly_charges: null, support_calls: 2, contract_type: 0, churn: 0 },
      { id: 7, tenure: 36, monthly_charges: 80, support_calls: 1, contract_type: 1, churn: 0 },
      { id: 8, tenure: 5, monthly_charges: 100, support_calls: 8, contract_type: 0, churn: null }
    ]
  }
};

export default function App() {
  const [selectedProj, setSelectedProj] = useState('housing'); // 'housing' | 'churn'
  const [currentStage, setCurrentStage] = useState(0); // 0 to 4
  const [completedStages, setCompletedStages] = useState([]);

  // Stage 1 (Obtain) State
  const [dataIngested, setDataIngested] = useState(false);
  const [rawDataset, setRawDataset] = useState([]);

  // Stage 2 (Scrub) State
  const [cleaningSteps, setCleaningSteps] = useState({
    imputeMissing: false,
    removeOutliers: false,
    scaleFeatures: false
  });
  const [cleanedDataset, setCleanedDataset] = useState([]);

  // Stage 3 (Explore) State
  const [edaXAxis, setEdaXAxis] = useState('');
  const [edaYAxis, setEdaYAxis] = useState('');
  const [hoveredPoint, setHoveredPoint] = useState(null);

  // Stage 4 (Model) State
  const [selectedModel, setSelectedModel] = useState('tree'); // 'tree' | 'linear'
  const [hyperparameters, setHyperparameters] = useState({
    learningRateOrDepth: 4, // Depth for Tree, learning rate (scaled) for linear
    epochsOrEstimators: 50
  });
  const [isTraining, setIsTraining] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [epochLogs, setEpochLogs] = useState([]);
  const [finalMetrics, setFinalMetrics] = useState(null);
  const [isModelTrained, setIsModelTrained] = useState(false);

  // Stage 5 (Interpret / Predict) State
  const [sliderInputs, setSliderInputs] = useState({});
  const [predictionResult, setPredictionResult] = useState(null);

  const datasetConfig = DATASETS[selectedProj];

  // Reset states when switching projects
  useEffect(() => {
    setDataIngested(false);
    setRawDataset([]);
    setCleaningSteps({ imputeMissing: false, removeOutliers: false, scaleFeatures: false });
    setCleanedDataset([]);
    setEdaXAxis(datasetConfig.features[0].id);
    setEdaYAxis(selectedProj === 'housing' ? 'price' : 'churn');
    setIsModelTrained(false);
    setFinalMetrics(null);
    setEpochLogs([]);
    setCurrentStage(0);
    setCompletedStages([]);
    
    // Initialize slider inputs for Stage 5 prediction
    const initialSliders = {};
    datasetConfig.features.forEach(f => {
      initialSliders[f.id] = f.default;
    });
    setSliderInputs(initialSliders);
  }, [selectedProj]);

  // Handle stage change
  const navigateStage = (index) => {
    if (index === 0 || completedStages.includes(index - 1) || index < currentStage) {
      setCurrentStage(index);
    }
  };

  const markStageComplete = (index) => {
    if (!completedStages.includes(index)) {
      setCompletedStages([...completedStages, index]);
    }
  };

  // Stage 1 Action: Ingest Data
  const handleIngestData = () => {
    setDataIngested(true);
    setRawDataset(datasetConfig.rawSample);
    markStageComplete(0);
  };

  // Stage 2 Action: Perform Data Cleaning
  const handleApplyCleaning = () => {
    let clean = JSON.parse(JSON.stringify(rawDataset));
    
    if (cleaningSteps.imputeMissing) {
      // Calculate simple averages to replace nulls
      clean = clean.map(row => {
        const imputedRow = { ...row };
        Object.keys(imputedRow).forEach(key => {
          if (imputedRow[key] === null) {
            // Pick a reasonable median/mean default
            if (key === 'sqft') imputedRow[key] = 2200;
            if (key === 'bedrooms') imputedRow[key] = 3;
            if (key === 'tenure') imputedRow[key] = 24;
            if (key === 'monthly_charges') imputedRow[key] = 70;
            if (key === 'price') imputedRow[key] = 390000;
            if (key === 'churn') imputedRow[key] = 0;
          }
        });
        return imputedRow;
      });
    }

    if (cleaningSteps.removeOutliers) {
      // Drop rows with extreme values or remaining un-imputed nulls
      clean = clean.filter(row => {
        if (!cleaningSteps.imputeMissing) {
          // If we haven't imputed, filter out nulls to protect down-stream modeling
          return !Object.values(row).includes(null);
        }
        return true;
      });
    }

    setCleanedDataset(clean);
    markStageComplete(1);
  };

  // Stage 4 Action: Trigger Training Simulation
  const handleTrainModel = () => {
    setIsTraining(true);
    setTrainingProgress(0);
    setEpochLogs([]);
    setIsModelTrained(false);

    let progress = 0;
    const interval = setInterval(() => {
      progress += 10;
      setTrainingProgress(progress);
      
      // Generate realistic logs
      const currentEpoch = progress / 10;
      const trainLoss = (1 / (currentEpoch + 1) * 0.5 + 0.1).toFixed(4);
      const valMetric = selectedProj === 'housing'
        ? (0.4 + (currentEpoch / 10) * 0.45 - (selectedModel === 'linear' ? 0.05 : 0)).toFixed(3)
        : (0.5 + (currentEpoch / 10) * 0.38 - (selectedModel === 'linear' ? 0.04 : 0)).toFixed(3);

      setEpochLogs(prev => [
        ...prev,
        `Epoch ${currentEpoch}/10 - Loss: ${trainLoss} - Val Metric: ${valMetric}`
      ]);

      if (progress >= 100) {
        clearInterval(interval);
        setIsTraining(false);
        setIsModelTrained(true);
        
        // Finalized metrics based on model configuration
        const accuracyMult = selectedModel === 'tree' ? 1.05 : 0.95;
        const finalVal = selectedProj === 'housing' 
          ? (0.83 * accuracyMult).toFixed(2) 
          : (0.88 * accuracyMult).toFixed(2);

        setFinalMetrics({
          valMetricValue: finalVal,
          trainTime: '1.4 seconds',
          featuresUsed: datasetConfig.features.length,
          modelType: selectedModel === 'tree' ? 'Gradient Boosted Trees' : 'Ridge Linear Classifier'
        });
        markStageComplete(3);
      }
    }, 150);
  };

  // Stage 5 Action: Run Live Prediction
  const runPrediction = () => {
    // Formula approximation for predictive display
    let score = 0;
    if (selectedProj === 'housing') {
      const basePrice = 120000;
      const sqftVal = (sliderInputs.sqft || 2000) * 150;
      const bedroomVal = (sliderInputs.bedrooms || 3) * 25000;
      const agePenalty = (sliderInputs.age || 10) * 1800;
      const locationBonus = (sliderInputs.location_score || 5) * 45000;
      const modelVariance = selectedModel === 'tree' ? 1.02 : 0.98; // simulated prediction variance

      score = (basePrice + sqftVal + bedroomVal - agePenalty + locationBonus) * modelVariance;
      setPredictionResult({
        primary: `$${Math.round(score).toLocaleString()}`,
        label: "Estimated Market Value",
        importance: [
          { name: 'Square Footage', val: 55, color: 'bg-emerald-500' },
          { name: 'Location Grade', val: 30, color: 'bg-indigo-500' },
          { name: 'Bedrooms', val: 10, color: 'bg-amber-500' },
          { name: 'Property Age', val: 5, color: 'bg-rose-500' }
        ]
      });
    } else {
      // Churn logic
      const tenurePenalty = (72 - (sliderInputs.tenure || 12)) * 1.5;
      const chargePenalty = (sliderInputs.monthly_charges || 70) * 0.8;
      const supportPenalty = (sliderInputs.support_calls || 2) * 15;
      const contractBonus = (sliderInputs.contract_type || 0) * 45;

      const rawProbability = Math.max(5, Math.min(98, tenurePenalty + chargePenalty + supportPenalty - contractBonus));
      
      setPredictionResult({
        primary: `${Math.round(rawProbability)}%`,
        label: "Churn Risk Score",
        riskLevel: rawProbability > 60 ? 'High Risk' : rawProbability > 30 ? 'Moderate Risk' : 'Low Risk',
        importance: [
          { name: 'Contract Type', val: 40, color: 'bg-indigo-500' },
          { name: 'Support Calls', val: 35, color: 'bg-rose-500' },
          { name: 'Tenure Length', val: 15, color: 'bg-amber-500' },
          { name: 'Monthly Bill', val: 10, color: 'bg-emerald-500' }
        ]
      });
    }
    markStageComplete(4);
  };

  useEffect(() => {
    if (isModelTrained) {
      runPrediction();
    }
  }, [sliderInputs, isModelTrained, selectedProj]);

  // Stage Meta info helper
  const stagesMeta = [
    { name: 'Obtain Data', icon: Database, desc: 'Ingestion & Sourcing' },
    { name: 'Scrub Data', icon: Trash2, desc: 'Cleaning & Formatting' },
    { name: 'Explore (EDA)', icon: BarChart3, desc: 'Patterns & Correlations' },
    { name: 'Model Training', icon: Brain, desc: 'Training & Tuning' },
    { name: 'Interpret Model', icon: Sliders, desc: 'Inference & Explainability' }
  ];

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-indigo-500/30 selection:text-indigo-200">
      
      {/* HEADER BAR */}
      <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur-md sticky top-0 z-50 px-6 py-4 flex flex-wrap justify-between items-center gap-4">
        <div className="flex items-center space-x-3">
          <div className="bg-gradient-to-tr from-indigo-500 to-violet-600 p-2.5 rounded-xl shadow-lg shadow-indigo-500/20">
            <TrendingUp className="h-6 w-6 text-white" />
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tight bg-gradient-to-r from-white via-slate-100 to-indigo-300 bg-clip-text text-transparent">
              ML Lifecycle Sandbox
            </h1>
            <p className="text-xs text-slate-400">Interactive End-to-End Data Science Simulator</p>
          </div>
        </div>

        {/* Project Selector Toggle */}
        <div className="flex bg-slate-950 p-1 rounded-xl border border-slate-800">
          <button
            onClick={() => setSelectedProj('housing')}
            className={`px-4 py-2 text-xs font-semibold rounded-lg transition-all duration-200 flex items-center space-x-2 ${
              selectedProj === 'housing'
                ? 'bg-gradient-to-r from-indigo-600 to-violet-600 text-white shadow-md shadow-indigo-600/10'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            <span>🏠 Real Estate Predictor</span>
          </button>
          <button
            onClick={() => setSelectedProj('churn')}
            className={`px-4 py-2 text-xs font-semibold rounded-lg transition-all duration-200 flex items-center space-x-2 ${
              selectedProj === 'churn'
                ? 'bg-gradient-to-r from-indigo-600 to-violet-600 text-white shadow-md shadow-indigo-600/10'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            <span>📈 Customer Churn Guard</span>
          </button>
        </div>
      </header>

      {/* STAGE STEPPER BAR */}
      <nav className="bg-slate-900 border-b border-slate-800 px-6 py-4 overflow-x-auto">
        <div className="max-w-7xl mx-auto flex items-center space-x-2 md:space-x-4 min-w-[760px]">
          {stagesMeta.map((stage, idx) => {
            const Icon = stage.icon;
            const isActive = currentStage === idx;
            const isCompleted = completedStages.includes(idx);
            const isSelectable = idx === 0 || completedStages.includes(idx - 1);

            return (
              <React.Fragment key={idx}>
                {idx > 0 && (
                  <ChevronRight className={`h-5 w-5 flex-shrink-0 ${isCompleted ? 'text-indigo-500' : 'text-slate-700'}`} />
                )}
                <button
                  disabled={!isSelectable}
                  onClick={() => navigateStage(idx)}
                  className={`flex-1 text-left p-3 rounded-xl transition-all duration-300 border focus:outline-none ${
                    isActive 
                      ? 'bg-slate-800 border-indigo-500 ring-1 ring-indigo-500/30' 
                      : isCompleted 
                        ? 'bg-slate-900/50 border-emerald-950 hover:border-slate-700' 
                        : isSelectable
                          ? 'bg-slate-900/30 border-slate-800 hover:border-slate-700'
                          : 'bg-slate-950/20 border-transparent opacity-40 cursor-not-allowed'
                  }`}
                >
                  <div className="flex items-center space-x-3">
                    <div className={`p-2 rounded-lg ${
                      isActive 
                        ? 'bg-indigo-600 text-white' 
                        : isCompleted 
                          ? 'bg-emerald-900/30 text-emerald-400' 
                          : 'bg-slate-800 text-slate-400'
                    }`}>
                      <Icon className="h-4 w-4" />
                    </div>
                    <div className="min-w-0">
                      <div className="flex items-center space-x-1.5">
                        <span className="text-[10px] uppercase tracking-wider font-semibold text-slate-500">
                          Stage 0{idx + 1}
                        </span>
                        {isCompleted && (
                          <span className="w-1.5 h-1.5 bg-emerald-400 rounded-full" />
                        )}
                      </div>
                      <h3 className="text-sm font-semibold truncate text-slate-200">
                        {stage.name}
                      </h3>
                    </div>
                  </div>
                </button>
              </React.Fragment>
            );
          })}
        </div>
      </nav>

      {/* MAIN CONTAINER */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 md:p-6 grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* LEFT COLUMN: INTERACTIVE LABORATORY WORKSPACE (8 cols) */}
        <section className="lg:col-span-8 flex flex-col gap-6">
          
          {/* STAGE COMPILER SCREEN */}
          <div className="bg-slate-900 rounded-2xl border border-slate-800 shadow-xl overflow-hidden flex-1 flex flex-col min-h-[500px]">
            
            {/* Stage Title Header */}
            <div className="bg-slate-800/40 px-6 py-4 border-b border-slate-800/80 flex justify-between items-center">
              <div className="flex items-center space-x-3">
                <span className="px-2.5 py-1 text-xs font-bold bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 rounded-md">
                  Active Workspace
                </span>
                <h2 className="text-lg font-bold text-slate-100">
                  {stagesMeta[currentStage].name}
                </h2>
              </div>
              <span className="text-xs text-slate-400">
                Project: <strong className="text-slate-200">{datasetConfig.name}</strong>
              </span>
            </div>

            {/* STAGE CONTROLLER CONTENT */}
            <div className="p-6 flex-1 flex flex-col justify-between">
              
              {/* STAGE 1: OBTAIN */}
              {currentStage === 0 && (
                <div className="space-y-6 flex-1 flex flex-col justify-between">
                  <div className="space-y-4">
                    <div className="bg-slate-950 p-4 rounded-xl border border-slate-800">
                      <h4 className="text-sm font-semibold text-slate-200 mb-1">Target Scenario & Business KPI</h4>
                      <p className="text-xs text-slate-400 leading-relaxed">
                        {datasetConfig.objective} We must source structural and demographic fields in order to construct our analytical dataset.
                      </p>
                    </div>

                    {!dataIngested ? (
                      <div className="flex flex-col items-center justify-center py-12 text-center space-y-4 border border-dashed border-slate-800 rounded-xl bg-slate-950/40">
                        <Database className="h-12 w-12 text-slate-600 animate-pulse" />
                        <div>
                          <h4 className="text-base font-semibold text-slate-300">Data Sourcing Required</h4>
                          <p className="text-xs text-slate-500 max-w-sm mx-auto mt-1">
                            Simulate pulling raw data from production log streams, client CRM tables, and flat files.
                          </p>
                        </div>
                        <button
                          onClick={handleIngestData}
                          className="px-6 py-2.5 bg-indigo-600 hover:bg-indigo-500 text-white rounded-xl text-xs font-bold flex items-center space-x-2 transition-all duration-200 shadow-lg shadow-indigo-600/20"
                        >
                          <Play className="h-3.5 w-3.5 fill-current" />
                          <span>Ingest Raw Logs & Databases</span>
                        </button>
                      </div>
                    ) : (
                      <div className="space-y-3">
                        <div className="flex justify-between items-center">
                          <span className="text-xs font-semibold text-slate-400">Raw Ingested Data Feed:</span>
                          <span className="text-[10px] bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 px-2 py-0.5 rounded-full font-mono">
                            STATUS: INGESTION SUCCESSFUL (8 Rows Cached)
                          </span>
                        </div>
                        <div className="overflow-x-auto border border-slate-800 rounded-xl bg-slate-950">
                          <table className="w-full text-left border-collapse text-xs">
                            <thead>
                              <tr className="bg-slate-900 border-b border-slate-800 text-slate-400">
                                <th className="p-3">ID</th>
                                {datasetConfig.features.map(f => (
                                  <th key={f.id} className="p-3">{f.name}</th>
                                ))}
                                <th className="p-3 text-indigo-400">{selectedProj === 'housing' ? 'Price (Label)' : 'Churn (Label)'}</th>
                              </tr>
                            </thead>
                            <tbody className="divide-y divide-slate-900 font-mono text-slate-300">
                              {rawDataset.map(row => (
                                <tr key={row.id} className="hover:bg-slate-900/40">
                                  <td className="p-3 text-slate-500">#{row.id}</td>
                                  {datasetConfig.features.map(f => (
                                    <td key={f.id} className="p-3">
                                      {row[f.id] === null ? (
                                        <span className="text-rose-500 font-bold bg-rose-500/10 px-1 py-0.5 rounded text-[10px]">NULL</span>
                                      ) : row[f.id]}
                                    </td>
                                  ))}
                                  <td className="p-3">
                                    {selectedProj === 'housing' ? (
                                      row.price === null ? (
                                        <span className="text-rose-500 font-bold bg-rose-500/10 px-1 py-0.5 rounded text-[10px]">NULL</span>
                                      ) : `$${row.price.toLocaleString()}`
                                    ) : (
                                      row.churn === null ? (
                                        <span className="text-rose-500 font-bold bg-rose-500/10 px-1 py-0.5 rounded text-[10px]">NULL</span>
                                      ) : row.churn === 1 ? 'Yes (1)' : 'No (0)'
                                    )}
                                  </td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                        <div className="flex items-start space-x-2 bg-amber-500/5 border border-amber-500/20 p-3 rounded-lg text-xs text-amber-300">
                          <AlertCircle className="h-4 w-4 flex-shrink-0 mt-0.5" />
                          <span>
                            <strong>Data Quality Alert:</strong> We have detected missing values (<span className="font-mono text-rose-400">NULL</span>) in features like <em>{selectedProj === 'housing' ? 'Square Footage & Bedrooms' : 'Tenure & Bill Amount'}</em>. Proceed to Stage 2 to scrub and fix this data.
                          </span>
                        </div>
                      </div>
                    )}
                  </div>

                  {dataIngested && (
                    <div className="border-t border-slate-800 pt-4 flex justify-end">
                      <button
                        onClick={() => navigateStage(1)}
                        className="px-5 py-2.5 bg-indigo-600 hover:bg-indigo-500 text-white rounded-xl text-xs font-semibold flex items-center space-x-2 shadow-lg shadow-indigo-600/15"
                      >
                        <span>Proceed to Scrub (Data Cleaning)</span>
                        <ChevronRight className="h-4 w-4" />
                      </button>
                    </div>
                  )}
                </div>
              )}

              {/* STAGE 2: SCRUB */}
              {currentStage === 1 && (
                <div className="space-y-6 flex-1 flex flex-col justify-between">
                  <div className="space-y-4">
                    <p className="text-xs text-slate-400">
                      Gbage in, garbage out. Configure preprocessing algorithms below to deal with missing records, scale numerical variables to equal bounds, and standardise formatting.
                    </p>

                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      {/* Step 1: Imputation */}
                      <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 flex flex-col justify-between">
                        <div>
                          <div className="flex justify-between items-start mb-2">
                            <span className="text-xs font-semibold text-slate-300">Mean/Median Imputation</span>
                            <input
                              type="checkbox"
                              checked={cleaningSteps.imputeMissing}
                              onChange={(e) => setCleaningSteps({ ...cleaningSteps, imputeMissing: e.target.checked })}
                              className="rounded bg-slate-900 border-slate-700 text-indigo-500 focus:ring-indigo-500 h-4 w-4"
                            />
                          </div>
                          <p className="text-[11px] text-slate-500">
                            Fills null values with statistical aggregates to prevent analytical model crash down.
                          </p>
                        </div>
                        <span className="text-[10px] mt-3 font-semibold text-indigo-400">Recommended</span>
                      </div>

                      {/* Step 2: Remove remaining bad rows */}
                      <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 flex flex-col justify-between">
                        <div>
                          <div className="flex justify-between items-start mb-2">
                            <span className="text-xs font-semibold text-slate-300">Target Leakage Protection</span>
                            <input
                              type="checkbox"
                              checked={cleaningSteps.removeOutliers}
                              onChange={(e) => setCleaningSteps({ ...cleaningSteps, removeOutliers: e.target.checked })}
                              className="rounded bg-slate-900 border-slate-700 text-indigo-500 focus:ring-indigo-500 h-4 w-4"
                            />
                          </div>
                          <p className="text-[11px] text-slate-500">
                            Strips rows where target variables or labels are missing or fall under corrupted indices.
                          </p>
                        </div>
                        <span className="text-[10px] mt-3 font-semibold text-indigo-400 font-mono">Filter Method</span>
                      </div>

                      {/* Step 3: Feature Scaling */}
                      <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 flex flex-col justify-between">
                        <div>
                          <div className="flex justify-between items-start mb-2">
                            <span className="text-xs font-semibold text-slate-300">Continuous Scaling</span>
                            <input
                              type="checkbox"
                              disabled
                              checked={true}
                              className="rounded bg-slate-900 border-slate-700 text-slate-500 cursor-not-allowed h-4 w-4"
                            />
                          </div>
                          <p className="text-[11px] text-slate-500">
                            Auto-normalizes continuous ranges into standard scales (z-score scaling) for algorithm stability.
                          </p>
                        </div>
                        <span className="text-[10px] mt-3 font-semibold text-slate-500 font-mono">Always Enabled</span>
                      </div>
                    </div>

                    <div className="flex justify-center py-2">
                      <button
                        onClick={handleApplyCleaning}
                        className="px-5 py-2.5 bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-bold rounded-lg flex items-center space-x-2 transition-all duration-200 shadow"
                      >
                        <RefreshCw className="h-3.5 w-3.5" />
                        <span>Apply Transforms & Standardise</span>
                      </button>
                    </div>

                    {cleanedDataset.length > 0 && (
                      <div className="space-y-3">
                        <h4 className="text-xs font-semibold text-slate-400">Post-Scrub Cleaned Matrix Preview:</h4>
                        <div className="overflow-x-auto border border-slate-800 rounded-xl bg-slate-950">
                          <table className="w-full text-left border-collapse text-xs">
                            <thead>
                              <tr className="bg-slate-900 border-b border-slate-800 text-slate-400">
                                <th className="p-3">ID</th>
                                {datasetConfig.features.map(f => (
                                  <th key={f.id} className="p-3">{f.name}</th>
                                ))}
                                <th className="p-3 text-indigo-400">{selectedProj === 'housing' ? 'Price' : 'Churn'}</th>
                              </tr>
                            </thead>
                            <tbody className="divide-y divide-slate-900 font-mono text-slate-300">
                              {cleanedDataset.map(row => (
                                <tr key={row.id} className="hover:bg-slate-900/40">
                                  <td className="p-3 text-slate-500">#{row.id}</td>
                                  {datasetConfig.features.map(f => (
                                    <td key={f.id} className="p-3 text-emerald-400">
                                      {row[f.id]}
                                    </td>
                                  ))}
                                  <td className="p-3 text-indigo-400">
                                    {selectedProj === 'housing' ? `$${row.price.toLocaleString()}` : row.churn === 1 ? 'Yes (1)' : 'No (0)'}
                                  </td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </div>
                    )}
                  </div>

                  {cleanedDataset.length > 0 && (
                    <div className="border-t border-slate-800 pt-4 flex justify-end">
                      <button
                        onClick={() => navigateStage(2)}
                        className="px-5 py-2.5 bg-indigo-600 hover:bg-indigo-500 text-white rounded-xl text-xs font-semibold flex items-center space-x-2 shadow-lg shadow-indigo-600/15"
                      >
                        <span>Proceed to Explore (EDA)</span>
                        <ChevronRight className="h-4 w-4" />
                      </button>
                    </div>
                  )}
                </div>
              )}

              {/* STAGE 3: EXPLORE */}
              {currentStage === 2 && (
                <div className="space-y-6 flex-1 flex flex-col justify-between">
                  <div className="space-y-4">
                    <p className="text-xs text-slate-400">
                      Formulate hypotheses, find correlations, and look for clustering using interactive feature plotting tools.
                    </p>

                    {/* AXIS CONTROLLERS */}
                    <div className="flex flex-wrap gap-4 p-4 bg-slate-950 rounded-xl border border-slate-800">
                      <div className="flex-1 min-w-[200px]">
                        <label className="block text-[11px] font-semibold text-slate-400 mb-1">X-Axis Variable (Feature)</label>
                        <select
                          value={edaXAxis}
                          onChange={(e) => setEdaXAxis(e.target.value)}
                          className="w-full bg-slate-900 border-slate-700 text-slate-300 rounded-lg text-xs p-2 focus:ring-indigo-500 focus:border-indigo-500"
                        >
                          {datasetConfig.features.map(f => (
                            <option key={f.id} value={f.id}>{f.name}</option>
                          ))}
                        </select>
                      </div>

                      <div className="flex-1 min-w-[200px]">
                        <label className="block text-[11px] font-semibold text-slate-400 mb-1">Y-Axis Variable (Target Label)</label>
                        <select
                          value={edaYAxis}
                          disabled
                          className="w-full bg-slate-900/50 border-slate-800 text-slate-500 rounded-lg text-xs p-2 cursor-not-allowed"
                        >
                          <option value={selectedProj === 'housing' ? 'price' : 'churn'}>
                            {selectedProj === 'housing' ? 'Price ($)' : 'Churn Outcome (0/1)'}
                          </option>
                        </select>
                      </div>
                    </div>

                    {/* INTERACTIVE SCATTER PLOT OR DISTRIBUTION VIEW */}
                    <div className="relative bg-slate-950/70 border border-slate-800 rounded-xl p-6 h-64 flex flex-col justify-between">
                      
                      {/* Grid representation */}
                      <div className="absolute inset-x-6 inset-y-12 grid grid-cols-4 grid-rows-4 pointer-events-none">
                        {[...Array(16)].map((_, i) => (
                          <div key={i} className="border-t border-l border-slate-900/60" />
                        ))}
                      </div>

                      {/* Title of Plot */}
                      <div className="text-center text-xs text-slate-400 z-10 font-medium">
                        Plotting <span className="text-indigo-400 font-bold">{datasetConfig.features.find(f => f.id === edaXAxis)?.name}</span> vs. <span className="text-indigo-400 font-bold">{selectedProj === 'housing' ? 'Price' : 'Churn Risk'}</span>
                      </div>

                      {/* Plot Content (SVG Based for high fidelity) */}
                      <div className="relative w-full h-40 mt-4 flex items-end justify-between px-6">
                        
                        {/* Render Plot Points */}
                        {cleanedDataset.length > 0 ? (
                          cleanedDataset.map((row, idx) => {
                            // Calculate simple percentages for rendering coordinates inside the SVG plane
                            const fMeta = datasetConfig.features.find(f => f.id === edaXAxis);
                            const xVal = row[edaXAxis];
                            const xPercent = ((xVal - fMeta.min) / (fMeta.max - fMeta.min)) * 85 + 5; // bounds limit 5% to 90%

                            let yPercent = 50;
                            if (selectedProj === 'housing') {
                              yPercent = ((row.price - 100000) / (900000 - 100000)) * 80 + 10;
                            } else {
                              yPercent = row.churn === 1 ? 85 : 15;
                            }

                            return (
                              <div
                                key={row.id}
                                className="absolute group cursor-pointer transition-all duration-300 transform -translate-x-1/2 translate-y-1/2"
                                style={{
                                  left: `${xPercent}%`,
                                  bottom: `${yPercent}%`
                                }}
                                onMouseEnter={() => setHoveredPoint(row)}
                                onMouseLeave={() => setHoveredPoint(null)}
                              >
                                <div className={`h-4.5 w-4.5 rounded-full border-2 bg-slate-900 flex items-center justify-center transition-all ${
                                  row.churn === 1 || (selectedProj === 'housing' && row.price > 500000)
                                    ? 'border-rose-500 hover:bg-rose-500/20'
                                    : 'border-emerald-500 hover:bg-emerald-500/20'
                                }`}>
                                  <div className={`h-1.5 w-1.5 rounded-full ${
                                    row.churn === 1 || (selectedProj === 'housing' && row.price > 500000)
                                      ? 'bg-rose-500'
                                      : 'bg-emerald-500'
                                  }`} />
                                </div>
                                
                                {/* Absolute tooltip on hover */}
                                {hoveredPoint?.id === row.id && (
                                  <div className="absolute bottom-6 left-1/2 transform -translate-x-1/2 bg-slate-900 border border-slate-700 p-2 rounded-lg text-[10px] whitespace-nowrap z-50 shadow-xl">
                                    <p className="font-semibold text-slate-100">ID #{row.id}</p>
                                    <p className="text-slate-400">{fMeta.name}: <span className="text-white font-mono">{row[edaXAxis]} {fMeta.unit}</span></p>
                                    <p className="text-slate-400">Target: <span className="text-white font-mono">
                                      {selectedProj === 'housing' ? `$${row.price.toLocaleString()}` : row.churn === 1 ? 'Churned (1)' : 'Active (0)'}
                                    </span></p>
                                  </div>
                                )}
                              </div>
                            );
                          })
                        ) : (
                          <div className="w-full text-center text-xs text-slate-600 py-10">
                            Apply data scrubbing transforms to unlock exploration view!
                          </div>
                        )}
                        
                        {/* Axes lines */}
                        <div className="absolute bottom-0 left-4 right-4 h-0.5 bg-slate-800" />
                        <div className="absolute top-2 bottom-0 left-4 w-0.5 bg-slate-800" />
                      </div>

                      {/* Axis Labeling Footers */}
                      <div className="flex justify-between items-center text-[10px] text-slate-500 px-2 mt-1">
                        <span>Low {datasetConfig.features.find(f => f.id === edaXAxis)?.name}</span>
                        <span>High {datasetConfig.features.find(f => f.id === edaXAxis)?.name}</span>
                      </div>
                    </div>

                    {/* STATISTICAL INSIGHT */}
                    <div className="bg-slate-900 border border-indigo-950/60 p-4 rounded-xl flex items-start space-x-3">
                      <Sparkles className="h-5 w-5 text-indigo-400 flex-shrink-0 mt-0.5" />
                      <div>
                        <h4 className="text-xs font-bold text-indigo-300">Auto Correlation Report</h4>
                        <p className="text-[11px] text-slate-400 leading-relaxed mt-1">
                          Our continuous bivariate analysis shows a {selectedProj === 'housing' ? 'strong positive linear correlation' : 'clear inverse log-odds pattern'} between standard factors and outcome profiles. This confirms structural signal exists—proceed with training models.
                        </p>
                      </div>
                    </div>
                  </div>

                  <div className="border-t border-slate-800 pt-4 flex justify-between">
                    <button
                      onClick={() => navigateStage(1)}
                      className="text-slate-400 hover:text-white text-xs font-semibold"
                    >
                      Back to Scrub
                    </button>
                    <button
                      onClick={() => {
                        markStageComplete(2);
                        navigateStage(3);
                      }}
                      className="px-5 py-2.5 bg-indigo-600 hover:bg-indigo-500 text-white rounded-xl text-xs font-semibold flex items-center space-x-2 shadow-lg shadow-indigo-600/15"
                    >
                      <span>Proceed to Model Training</span>
                      <ChevronRight className="h-4 w-4" />
                    </button>
                  </div>
                </div>
              )}

              {/* STAGE 4: MODEL */}
              {currentStage === 3 && (
                <div className="space-y-6 flex-1 flex flex-col justify-between">
                  <div className="space-y-4">
                    <p className="text-xs text-slate-400">
                      Configure your machine learning algorithm, calibrate hyper-parameters, and execute the backpropagation training loop in standard memory blocks.
                    </p>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {/* Left: Model Architecture Settings */}
                      <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-4">
                        <h4 className="text-xs font-bold text-slate-300 uppercase tracking-wide">Model Architecture</h4>
                        
                        <div className="space-y-2">
                          <button
                            onClick={() => setSelectedModel('tree')}
                            className={`w-full text-left p-3 rounded-lg border text-xs flex justify-between items-center transition-all ${
                              selectedModel === 'tree'
                                ? 'bg-indigo-950/40 border-indigo-500/80 text-white'
                                : 'bg-slate-900/30 border-slate-800 text-slate-400 hover:border-slate-700'
                            }`}
                          >
                            <div>
                              <p className="font-semibold">Tree-Based Ensemble Model</p>
                              <p className="text-[10px] text-slate-500">Robust with categorical and non-linear data structures.</p>
                            </div>
                            {selectedModel === 'tree' && <Check className="h-4 w-4 text-indigo-400" />}
                          </button>

                          <button
                            onClick={() => setSelectedModel('linear')}
                            className={`w-full text-left p-3 rounded-lg border text-xs flex justify-between items-center transition-all ${
                              selectedModel === 'linear'
                                ? 'bg-indigo-950/40 border-indigo-500/80 text-white'
                                : 'bg-slate-900/30 border-slate-800 text-slate-400 hover:border-slate-700'
                            }`}
                          >
                            <div>
                              <p className="font-semibold">Regularized Linear Predictor</p>
                              <p className="text-[10px] text-slate-500">High interpretability, quick training with minimal weights.</p>
                            </div>
                            {selectedModel === 'linear' && <Check className="h-4 w-4 text-indigo-400" />}
                          </button>
                        </div>

                        {/* Hyperparameter Adjustments */}
                        <div className="space-y-3 pt-2">
                          <h5 className="text-[11px] font-bold text-slate-400">Hyper-parameters Tuning</h5>
                          
                          {selectedModel === 'tree' ? (
                            <div>
                              <div className="flex justify-between text-[11px] text-slate-400 mb-1">
                                <span>Max Tree Depth</span>
                                <span className="font-mono text-indigo-400 font-bold">{hyperparameters.learningRateOrDepth} layers</span>
                              </div>
                              <input
                                type="range"
                                min="2"
                                max="10"
                                value={hyperparameters.learningRateOrDepth}
                                onChange={(e) => setHyperparameters({ ...hyperparameters, learningRateOrDepth: parseInt(e.target.value) })}
                                className="w-full accent-indigo-500"
                              />
                            </div>
                          ) : (
                            <div>
                              <div className="flex justify-between text-[11px] text-slate-400 mb-1">
                                <span>Learning Rate Alpha</span>
                                <span className="font-mono text-indigo-400 font-bold">{(hyperparameters.learningRateOrDepth * 0.025).toFixed(3)}</span>
                              </div>
                              <input
                                type="range"
                                min="1"
                                max="10"
                                value={hyperparameters.learningRateOrDepth}
                                onChange={(e) => setHyperparameters({ ...hyperparameters, learningRateOrDepth: parseInt(e.target.value) })}
                                className="w-full accent-indigo-500"
                              />
                            </div>
                          )}
                        </div>
                      </div>

                      {/* Right: Simulated Terminal Output & Logs */}
                      <div className="bg-slate-950 rounded-xl border border-slate-800 flex flex-col overflow-hidden font-mono text-[11px] h-60">
                        <div className="bg-slate-900 px-3 py-2 border-b border-slate-800 flex items-center justify-between text-slate-400">
                          <span>CONSOLE CONTEXT LOGS</span>
                          <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                        </div>
                        <div className="p-3 flex-1 overflow-y-auto space-y-1 text-slate-400 scrollbar-thin">
                          <div>&gt; Initializing backpropagation layers...</div>
                          {epochLogs.map((log, lidx) => (
                            <div key={lidx} className="text-slate-300">{`> ${log}`}</div>
                          ))}
                          {isTraining && (
                            <div className="text-indigo-400 animate-pulse">&gt; Calculating gradients and weights...</div>
                          )}
                          {isModelTrained && (
                            <div className="text-emerald-400 font-bold">&gt; Target convergence met. Training complete!</div>
                          )}
                        </div>
                      </div>
                    </div>

                    {/* TRAINING BUTTON */}
                    <div className="flex justify-center">
                      <button
                        onClick={handleTrainModel}
                        disabled={isTraining}
                        className={`px-8 py-3 rounded-xl text-xs font-bold flex items-center space-x-2 transition-all duration-300 ${
                          isTraining
                            ? 'bg-slate-800 text-slate-500 cursor-not-allowed'
                            : 'bg-gradient-to-r from-indigo-600 to-violet-600 hover:from-indigo-500 hover:to-violet-500 text-white shadow-lg shadow-indigo-600/20'
                        }`}
                      >
                        {isTraining ? (
                          <>
                            <RefreshCw className="h-4 w-4 animate-spin text-indigo-400" />
                            <span>Optimizing Weights ({trainingProgress}%)</span>
                          </>
                        ) : (
                          <>
                            <Brain className="h-4 w-4" />
                            <span>Start Train Loop</span>
                          </>
                        )}
                      </button>
                    </div>

                    {/* METRIC CARD ONCE TRAINED */}
                    {finalMetrics && (
                      <div className="bg-emerald-950/20 border border-emerald-900/40 p-4 rounded-xl grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
                        <div>
                          <p className="text-[10px] text-slate-400 uppercase font-semibold">Trained Model</p>
                          <p className="text-xs font-bold text-white mt-0.5">{finalMetrics.modelType}</p>
                        </div>
                        <div>
                          <p className="text-[10px] text-slate-400 uppercase font-semibold">Validation Metric</p>
                          <p className="text-xs font-bold text-emerald-400 mt-0.5">{datasetConfig.metricName}: {finalMetrics.valMetricValue}</p>
                        </div>
                        <div>
                          <p className="text-[10px] text-slate-400 uppercase font-semibold">Execution Speed</p>
                          <p className="text-xs font-bold text-white mt-0.5">{finalMetrics.trainTime}</p>
                        </div>
                        <div>
                          <p className="text-[10px] text-slate-400 uppercase font-semibold">Features Loaded</p>
                          <p className="text-xs font-bold text-white mt-0.5">{finalMetrics.featuresUsed} features</p>
                        </div>
                      </div>
                    )}
                  </div>

                  <div className="border-t border-slate-800 pt-4 flex justify-between">
                    <button
                      onClick={() => navigateStage(2)}
                      className="text-slate-400 hover:text-white text-xs font-semibold"
                    >
                      Back to Explore
                    </button>
                    <button
                      disabled={!isModelTrained}
                      onClick={() => navigateStage(4)}
                      className={`px-5 py-2.5 rounded-xl text-xs font-semibold flex items-center space-x-2 transition-all ${
                        isModelTrained
                          ? 'bg-indigo-600 hover:bg-indigo-500 text-white shadow-lg shadow-indigo-600/15'
                          : 'bg-slate-800 text-slate-500 cursor-not-allowed'
                      }`}
                    >
                      <span>Proceed to Interpret Model</span>
                      <ChevronRight className="h-4 w-4" />
                    </button>
                  </div>
                </div>
              )}

              {/* STAGE 5: INTERPRET */}
              {currentStage === 4 && (
                <div className="space-y-6 flex-1 flex flex-col justify-between">
                  <div className="space-y-4">
                    <p className="text-xs text-slate-400">
                      Inspect model predictions via continuous user input parameters. Visualize the SHAP local feature importances below to explain <strong>why</strong> predictions occur.
                    </p>

                    {!isModelTrained ? (
                      <div className="flex flex-col items-center justify-center py-12 text-center space-y-3 bg-slate-950/40 border border-dashed border-slate-800 rounded-xl">
                        <AlertCircle className="h-10 w-10 text-amber-500" />
                        <div>
                          <h4 className="text-sm font-semibold text-slate-300">Model Training Required</h4>
                          <p className="text-xs text-slate-500 max-w-sm mt-1">
                            Go back to Stage 4 to train your model before simulating real-time inference scenarios.
                          </p>
                        </div>
                        <button
                          onClick={() => navigateStage(3)}
                          className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-bold rounded-lg transition-all"
                        >
                          Go Train Model
                        </button>
                      </div>
                    ) : (
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        
                        {/* Interactive Sliders for Prediction Inputs */}
                        <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-4">
                          <h4 className="text-xs font-bold text-slate-300 uppercase tracking-wide">Live Feature Input Values</h4>
                          
                          <div className="space-y-3">
                            {datasetConfig.features.map(f => (
                              <div key={f.id} className="space-y-1">
                                <div className="flex justify-between text-xs text-slate-400">
                                  <span>{f.name}</span>
                                  <span className="font-mono text-indigo-400 font-semibold">
                                    {sliderInputs[f.id]} {f.unit}
                                  </span>
                                </div>
                                <input
                                  type="range"
                                  min={f.min}
                                  max={f.max}
                                  value={sliderInputs[f.id] || f.default}
                                  onChange={(e) => setSliderInputs({ ...sliderInputs, [f.id]: parseFloat(e.target.value) })}
                                  className="w-full accent-indigo-500 h-1.5 rounded-lg bg-slate-900"
                                />
                              </div>
                            ))}
                          </div>
                        </div>

                        {/* Interactive Prediction Display & Attribution (SHAP) */}
                        <div className="space-y-4 flex flex-col justify-between">
                          
                          {/* Main Prediction Score Box */}
                          <div className="bg-slate-950 p-5 rounded-xl border border-slate-800 flex items-center justify-between">
                            <div>
                              <p className="text-[10px] text-slate-400 uppercase tracking-wider font-semibold">{predictionResult?.label}</p>
                              <p className="text-2xl font-bold bg-gradient-to-r from-emerald-400 to-indigo-400 bg-clip-text text-transparent mt-1">
                                {predictionResult?.primary}
                              </p>
                              {predictionResult?.riskLevel && (
                                <span className={`inline-block text-[10px] font-bold px-2 py-0.5 rounded-full mt-2 border ${
                                  predictionResult.riskLevel === 'High Risk'
                                    ? 'bg-rose-500/10 border-rose-500/20 text-rose-400'
                                    : predictionResult.riskLevel === 'Moderate Risk'
                                      ? 'bg-amber-500/10 border-amber-500/20 text-amber-400'
                                      : 'bg-emerald-500/10 border-emerald-500/20 text-emerald-400'
                                }`}>
                                  {predictionResult.riskLevel}
                                </span>
                              )}
                            </div>
                            <Server className="h-10 w-10 text-slate-700" />
                          </div>

                          {/* Local Attribution */}
                          <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-3">
                            <div className="flex justify-between items-center">
                              <h5 className="text-[11px] font-bold text-slate-300 uppercase tracking-wide">Feature Importance (SHAP Explainability)</h5>
                              <HelpCircle className="h-3.5 w-3.5 text-slate-500" />
                            </div>

                            <div className="space-y-2">
                              {predictionResult?.importance.map((imp, idx) => (
                                <div key={idx} className="space-y-1">
                                  <div className="flex justify-between text-[10px] text-slate-400">
                                    <span>{imp.name}</span>
                                    <span>{imp.val}% Contribution</span>
                                  </div>
                                  <div className="w-full bg-slate-900 h-1.5 rounded-full overflow-hidden">
                                    <div className={`h-full ${imp.color}`} style={{ width: `${imp.val}%` }} />
                                  </div>
                                </div>
                              ))}
                            </div>
                          </div>

                        </div>

                      </div>
                    )}
                  </div>

                  <div className="border-t border-slate-800 pt-4 flex justify-between">
                    <button
                      onClick={() => navigateStage(3)}
                      className="text-slate-400 hover:text-white text-xs font-semibold"
                    >
                      Back to Training
                    </button>
                    <button
                      onClick={() => {
                        setSelectedProj(selectedProj === 'housing' ? 'churn' : 'housing');
                      }}
                      className="px-5 py-2.5 bg-gradient-to-r from-emerald-600 to-indigo-600 text-white rounded-xl text-xs font-semibold flex items-center space-x-2 shadow-lg"
                    >
                      <span>Try the Other Scenario</span>
                      <RefreshCw className="h-3.5 w-3.5" />
                    </button>
                  </div>
                </div>
              )}

            </div>

          </div>
          
        </section>

        {/* RIGHT COLUMN: EDUCATIONAL REFERENCE MANUAL (4 cols) */}
        <section className="lg:col-span-4 space-y-6">
          
          {/* EDUCATIONAL BOOKLET CARD */}
          <div className="bg-slate-900 rounded-2xl border border-slate-800 p-6 space-y-6 shadow-xl sticky top-24">
            
            <div className="flex items-center space-x-2 pb-2 border-b border-slate-800">
              <Info className="h-5 w-5 text-indigo-400" />
              <h3 className="text-sm font-bold text-slate-100 uppercase tracking-wider">
                Lifecycle Companion
              </h3>
            </div>

            {/* STAGE-SPECIFIC DOCUMENTATION */}
            {currentStage === 0 && (
              <div className="space-y-4">
                <h4 className="text-sm font-semibold text-indigo-400">Stage 1: Obtain (Data Acquisition)</h4>
                <p className="text-xs text-slate-400 leading-relaxed">
                  Every data science cycle initiates with defining the business problem statement and identifying what data sources contain features (predictive variables) and target labels.
                </p>
                <div className="space-y-2 bg-slate-950 p-3 rounded-lg border border-slate-800">
                  <span className="text-[10px] font-bold text-indigo-300 block uppercase">Key Tasks & Tools</span>
                  <ul className="text-[11px] text-slate-400 list-disc list-inside space-y-1">
                    <li>SQL Databases queries & joins</li>
                    <li>S3/Cloud Blob storage pipelines</li>
                    <li>API integrations (REST, gRPC)</li>
                    <li>Target Label identification</li>
                  </ul>
                </div>
              </div>
            )}

            {currentStage === 1 && (
              <div className="space-y-4">
                <h4 className="text-sm font-semibold text-indigo-400">Stage 2: Scrub (Data Wrangling)</h4>
                <p className="text-xs text-slate-400 leading-relaxed">
                  Real-world data is dirty, broken, and contains incomplete cycles. Scrubbing filters outliers, handles structural null values, standardizes measurements, and prevents target leakage.
                </p>
                <div className="space-y-2 bg-slate-950 p-3 rounded-lg border border-slate-800">
                  <span className="text-[10px] font-bold text-indigo-300 block uppercase">Key Algorithms</span>
                  <ul className="text-[11px] text-slate-400 list-disc list-inside space-y-1">
                    <li>K-Nearest Neighbors Imputation</li>
                    <li>Median/Mean statistical replacements</li>
                    <li>One-Hot Encoding for text variables</li>
                    <li>Min-Max or Standard scaling</li>
                  </ul>
                </div>
              </div>
            )}

            {currentStage === 2 && (
              <div className="space-y-4">
                <h4 className="text-sm font-semibold text-indigo-400">Stage 3: Explore (EDA)</h4>
                <p className="text-xs text-slate-400 leading-relaxed">
                  Exploratory Data Analysis (EDA) allows humans to build physical intuition about variables. Looking for clustering, correlation, and signal prevents useless modeling later.
                </p>
                <div className="space-y-2 bg-slate-950 p-3 rounded-lg border border-slate-800">
                  <span className="text-[10px] font-bold text-indigo-300 block uppercase">Visual Mechanics</span>
                  <ul className="text-[11px] text-slate-400 list-disc list-inside space-y-1">
                    <li>Scatter plots to identify trends</li>
                    <li>Correlation Matrix heatmaps</li>
                    <li>Histograms for skewness tests</li>
                    <li>ANOVA & t-test validations</li>
                  </ul>
                </div>
              </div>
            )}

            {currentStage === 3 && (
              <div className="space-y-4">
                <h4 className="text-sm font-semibold text-indigo-400">Stage 4: Model (Machine Learning)</h4>
                <p className="text-xs text-slate-400 leading-relaxed">
                  Here, numerical features are run through mathematical objective functions to optimize weights. Tuning hyper-parameters prevents either underfitting or overfitting.
                </p>
                <div className="space-y-2 bg-slate-950 p-3 rounded-lg border border-slate-800">
                  <span className="text-[10px] font-bold text-indigo-300 block uppercase">Training KPIs</span>
                  <ul className="text-[11px] text-slate-400 list-disc list-inside space-y-1">
                    <li><strong>R² & RMSE:</strong> Used for regression</li>
                    <li><strong>F1 & AUC:</strong> Used for classification</li>
                    <li><strong>Loss Curves:</strong> Show training health</li>
                    <li>Cross-validation folds</li>
                  </ul>
                </div>
              </div>
            )}

            {currentStage === 4 && (
              <div className="space-y-4">
                <h4 className="text-sm font-semibold text-indigo-400">Stage 5: Interpret (Inference)</h4>
                <p className="text-xs text-slate-400 leading-relaxed">
                  A high-performing model is useless if stakeholders cannot understand or deploy it. Interpretability tools (like SHAP / LIME values) explain feature attributions clearly.
                </p>
                <div className="space-y-2 bg-slate-950 p-3 rounded-lg border border-slate-800">
                  <span className="text-[10px] font-bold text-indigo-300 block uppercase">Deployment Paradigms</span>
                  <ul className="text-[11px] text-slate-400 list-disc list-inside space-y-1">
                    <li>REST APIs (Docker + Kubernetes)</li>
                    <li>Serverless Cloud functions</li>
                    <li>Continuous Model monitoring (Drift)</li>
                    <li>SHAP values explanation</li>
                  </ul>
                </div>
              </div>
            )}

            {/* PIPELINE OVERVIEW STATUS */}
            <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-3">
              <span className="text-xs font-bold text-slate-300 block uppercase tracking-wider">
                Overall Pipeline Status
              </span>
              <div className="space-y-2">
                {stagesMeta.map((s, idx) => (
                  <div key={idx} className="flex items-center justify-between text-xs">
                    <span className="text-slate-400">{s.name}</span>
                    {completedStages.includes(idx) ? (
                      <span className="text-emerald-400 font-semibold flex items-center space-x-1">
                        <CheckCircle className="h-3 w-3" />
                        <span>Ready</span>
                      </span>
                    ) : (
                      <span className="text-slate-600">Pending</span>
                    )}
                  </div>
                ))}
              </div>
            </div>

          </div>

        </section>

      </main>

      {/* FOOTER */}
      <footer className="border-t border-slate-900 bg-slate-950 py-6 text-center text-xs text-slate-500">
        <p>© 2026 Machine Learning Lifecycle Sandbox. Built for educational excellence.</p>
      </footer>

    </div>
  );
}