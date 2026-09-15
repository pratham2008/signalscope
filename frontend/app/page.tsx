"use client";

import { ChangeEvent, DragEvent, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

type ExplanationEvidence = {
  base_probability: number;
  max_positive_influence: number;
  max_negative_influence: number;
  mean_absolute_influence: number;
};

type Prediction = {
  label: string;
  confidence: number;
  probability_ai_generated: number;
  threshold: number;
  model: string;
  explanation_image?: string;
  explanation_method?: string;
  explanation_evidence?: ExplanationEvidence;
  explanation_text?: string;
};

function UploadIcon() {
  return (
    <svg viewBox="0 0 24 24" className="w-5 h-5 fill-none stroke-current stroke-[1.65] stroke-linecap-round stroke-linejoin-round">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M17 8l-5-5-5 5M12 3v12" />
    </svg>
  );
}

function ImageIcon() {
  return (
    <svg viewBox="0 0 24 24" className="w-4 h-4 fill-none stroke-current stroke-[1.8]">
      <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
      <circle cx="8.5" cy="8.5" r="1.5" />
      <polyline points="21 15 16 10 5 21" />
    </svg>
  );
}

function ClearIcon() {
  return (
    <svg viewBox="0 0 24 24" className="w-4 h-4 fill-none stroke-current stroke-[1.8]">
      <path d="M18 6 6 18M6 6l12 12" />
    </svg>
  );
}

function PlaceholderIcon() {
  return (
    <svg viewBox="0 0 24 24" className="w-5 h-5 fill-none stroke-current stroke-[1.65] stroke-linecap-round stroke-linejoin-round">
      <circle cx="12" cy="12" r="10" />
      <path d="M12 16v-4M12 8h.01" />
    </svg>
  );
}

export default function Home() {
  const inputRef = useRef<HTMLInputElement>(null);

  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<Prediction | null>(null);

  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function selectFile(selectedFile: File | undefined) {
    if (!selectedFile) {
      return;
    }

    if (!selectedFile.type.startsWith("image/")) {
      setError("Please select a valid image file.");
      return;
    }

    setFile(selectedFile);
    setPrediction(null);
    setError(null);

    const url = URL.createObjectURL(selectedFile);
    setPreview(url);
  }

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    selectFile(event.target.files?.[0]);
  }

  function handleDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDragging(false);
    selectFile(event.dataTransfer.files?.[0]);
  }

  async function analyzeImage() {
    if (!file) return;

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Prediction failed.");
      }

      setPrediction(data);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Unable to connect to SignalScope."
      );
    } finally {
      setLoading(false);
    }
  }

  function clearImage() {
    if (preview) {
      URL.revokeObjectURL(preview);
    }

    setFile(null);
    setPreview(null);
    setPrediction(null);
    setError(null);

    if (inputRef.current) {
      inputRef.current.value = "";
    }
  }

  const isAi = prediction?.label === "Likely AI-Generated";

  return (
    <main className="min-h-screen bg-noise bg-mesh">
      <div className="max-w-[1440px] mx-auto px-5 lg:px-10 py-6 lg:py-8">
        <header className="flex items-center justify-between border-b border-border pb-4 mb-10">
          <div className="flex items-center gap-2.5 text-[15px] font-semibold tracking-tight text-foreground">
            <div className="grid w-[25px] h-[25px] place-items-center border border-border rounded-md bg-secondary">
              <span className="relative w-[9px] h-[9px] border-2 border-primary rounded-full after:content-[''] after:absolute after:w-[3px] after:h-[3px] after:bg-primary after:rounded-full after:top-[1px] after:right-[1px]" />
            </div>
            SignalScope
          </div>

          <Badge variant="secondary" className="gap-2 font-mono text-[10px] uppercase tracking-wider text-muted-foreground bg-transparent border-border">
            <div className="w-[7px] h-[7px] rounded-full bg-primary shadow-[0_0_0_3px_rgba(255,255,255,0.1)]" />
            CLIP + FFT
          </Badge>
        </header>

        <div className="flex flex-col lg:flex-row lg:items-end justify-between gap-6 lg:gap-8 pb-8">
          <div>
            <p className="m-0 text-muted-foreground font-mono text-[10px] font-semibold tracking-[0.13em] uppercase leading-tight">Generalization-first detection</p>
            <h1 className="max-w-[720px] mt-2 text-[31px] md:text-[clamp(31px,4vw,52px)] font-semibold tracking-[-0.05em] leading-[1.06]">
              Understand whether an image is likely real or AI-generated.
            </h1>
          </div>

          <p className="max-w-[390px] m-0 text-muted-foreground text-[14px] leading-[1.65]">
            Upload an image and SignalScope analyzes semantic and frequency
            evidence to estimate the likelihood that it was generated by AI.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-[1.08fr_0.92fr] gap-5 items-start mt-4">
          <Card className="border-border bg-card shadow-sm rounded-[14px]">
            <CardHeader className="flex flex-row items-start justify-between pb-4">
              <div>
                <p className="m-0 text-muted-foreground font-mono text-[10px] font-semibold tracking-[0.13em] uppercase">01 / Source</p>
                <CardTitle className="mt-1.5 text-[17px] font-semibold tracking-tight">Image Input</CardTitle>
              </div>
              {file && (
                <Button variant="ghost" size="sm" onClick={clearImage} className="text-muted-foreground h-8 px-2 text-xs flex gap-1">
                  <ClearIcon /> Clear
                </Button>
              )}
            </CardHeader>
            <CardContent>
              {!file ? (
                <div
                  className={`relative flex flex-col items-center justify-center w-full min-h-[300px] border border-dashed rounded-lg cursor-pointer transition-all duration-200 ${dragging ? "border-primary bg-secondary/50 scale-[0.997]" : "border-border bg-muted/20 hover:border-muted-foreground hover:bg-muted/30"}`}
                  onDragOver={(event) => {
                    event.preventDefault();
                    setDragging(true);
                  }}
                  onDragLeave={() => setDragging(false)}
                  onDrop={handleDrop}
                  onClick={() => inputRef.current?.click()}
                >
                  <div className="flex flex-col items-center gap-2 p-8 text-center text-[13px] text-muted-foreground">
                    <div className="grid w-10 h-10 place-items-center border border-border rounded-lg text-primary mb-1 bg-secondary/50">
                      <UploadIcon />
                    </div>
                    <div>
                      <strong className="text-foreground text-[14px] font-medium block">Drop an image here</strong>
                      <p>or click to browse</p>
                    </div>
                    <div className="mt-2 text-muted-foreground/70 text-[11px]">
                      JPEG, PNG, WebP, or BMP
                    </div>
                  </div>
                  <input
                    ref={inputRef}
                    type="file"
                    accept="image/jpeg,image/png,image/webp,image/bmp"
                    className="visually-hidden"
                    onChange={handleFileChange}
                  />
                </div>
              ) : (
                <div
                  className="relative flex flex-col items-center justify-center w-full min-h-[300px] border border-border rounded-lg cursor-pointer bg-black/40 overflow-hidden"
                  onClick={() => inputRef.current?.click()}
                >
                  <img src={preview!} alt="Selected image" className="block w-full h-[300px] object-contain" />
                  <input
                    ref={inputRef}
                    type="file"
                    accept="image/jpeg,image/png,image/webp,image/bmp"
                    className="visually-hidden"
                    onChange={handleFileChange}
                  />
                </div>
              )}

              {file && (
                <div className="flex items-center gap-2.5 min-h-[41px] mt-4 border-b border-border text-muted-foreground font-mono text-[10px] pb-3">
                  <div className="inline-flex items-center gap-1.5 text-primary">
                    <ImageIcon /> Image
                  </div>
                  <div className="min-w-0 overflow-hidden text-ellipsis whitespace-nowrap text-foreground/80">{file.name}</div>
                  <div className="ml-auto whitespace-nowrap hidden sm:block">{(file.size / 1024 / 1024).toFixed(2)} MB</div>
                </div>
              )}

              {error && (
                <Alert variant="destructive" className="mt-4 rounded-md bg-destructive/10 text-destructive border-l-2 border-l-destructive border-y-0 border-r-0">
                  <AlertDescription className="text-xs">{error}</AlertDescription>
                </Alert>
              )}

              <Button
                disabled={!file || loading}
                onClick={analyzeImage}
                className="w-full h-11 mt-5 font-bold tracking-tight text-[13px] bg-primary text-primary-foreground hover:bg-primary/90"
              >
                {loading ? "Analyzing image..." : "Analyze Image"}
              </Button>
            </CardContent>
          </Card>

          <Card className={`border-border bg-card shadow-sm rounded-[14px] flex flex-col ${prediction ? "border-muted-foreground/30" : ""}`}>
            <CardHeader className="flex flex-row items-start justify-between pb-4">
              <div>
                <p className="m-0 text-muted-foreground font-mono text-[10px] font-semibold tracking-[0.13em] uppercase">02 / Assessment</p>
                <CardTitle className="mt-1.5 text-[17px] font-semibold tracking-tight">Analysis Result</CardTitle>
              </div>
              <Badge variant="outline" className={`font-mono text-[10px] uppercase gap-1.5 px-2 py-0.5 rounded-full ${loading ? 'border-primary/30 text-primary' : prediction ? 'border-[#77d7ae]/30 text-[#b5e7d0]' : 'border-border text-muted-foreground'}`}>
                <span className={`w-1.5 h-1.5 rounded-full ${loading ? 'bg-primary animate-[blink_1.25s_ease-in-out_infinite]' : prediction ? 'bg-[#77d7ae]' : 'bg-muted-foreground'}`} />
                {loading ? "Analyzing" : prediction ? "Complete" : "Waiting"}
              </Badge>
            </CardHeader>

            <CardContent className="flex-1 flex flex-col">
              {loading ? (
                <div className="flex-1 grid place-content-center justify-items-center text-center gap-3 min-h-[300px]">
                  <div className="scanner">
                    <span />
                  </div>
                  <p className="text-muted-foreground text-sm max-w-[285px] leading-relaxed">Running the detector and generating model evidence...</p>
                </div>
              ) : !prediction ? (
                <div className="flex-1 grid place-content-center justify-items-start gap-3 min-h-[300px] text-muted-foreground">
                  <div className="grid w-10 h-10 place-items-center border border-border rounded-lg text-primary">
                    <PlaceholderIcon />
                  </div>
                  <p className="text-sm max-w-[285px] leading-relaxed m-0">
                    Upload an image and click “Analyze Image” to view the
                    detector&apos;s assessment.
                  </p>
                </div>
              ) : (
                <div className="assessment-content">
                  <div className="flex items-center gap-3 my-6">
                    <div className={`w-2.5 h-2.5 rounded-full shadow-[0_0_0_4px_rgba(255,255,255,0.05)] ${isAi ? 'bg-[var(--accent-warn)]' : 'bg-[var(--accent-good)]'}`} />
                    <h3 className={`m-0 text-2xl md:text-3xl font-semibold tracking-tight ${isAi ? 'text-destructive-foreground' : 'text-foreground'}`}>{prediction.label}</h3>
                  </div>

                  <div className="py-5 border-y border-border">
                    <div className="flex items-baseline justify-between gap-4 text-[13px] text-muted-foreground">
                      <span>P(AI-generated)</span>
                      <strong className="text-3xl md:text-4xl font-semibold tracking-tight text-foreground">
                        {(prediction.probability_ai_generated * 100).toFixed(2)}%
                      </strong>
                    </div>

                    <div className="probability-scale">
                      <div
                        className={`probability-fill ${isAi ? "bg-[var(--accent-warn)]" : "bg-[var(--accent-good)]"}`}
                        style={{
                          width: `${prediction.probability_ai_generated * 100}%`,
                        }}
                      />
                      <div
                        className="threshold-marker"
                        style={{ left: `${prediction.threshold * 100}%` }}
                      >
                        <i />
                        <b>Threshold {(prediction.threshold * 100).toFixed(2)}%</b>
                      </div>
                    </div>

                    <div className="flex justify-between mt-6 text-[10px] text-muted-foreground">
                      <span>Likely real</span>
                      <span>Likely AI</span>
                    </div>
                  </div>

                  <dl className="grid grid-cols-2 gap-4 mt-5">
                    <div className="border-l border-border pl-3">
                      <dt className="text-[10px] text-muted-foreground">Confidence</dt>
                      <dd className="mt-1 font-mono text-[14px] text-foreground">{(prediction.confidence * 100).toFixed(2)}%</dd>
                    </div>
                    <div className="border-l border-border pl-3">
                      <dt className="text-[10px] text-muted-foreground">Model Engine</dt>
                      <dd className="mt-1 font-mono text-[14px] text-foreground">{prediction.model}</dd>
                    </div>
                  </dl>

                  {prediction.explanation_image && (
                    <div className="evidence-section mt-5 pt-6 border-t border-border">
                      <div className="flex flex-col md:flex-row md:items-end justify-between gap-4 mb-4">
                        <div>
                          <p className="m-0 text-muted-foreground font-mono text-[10px] font-semibold tracking-[0.13em] uppercase">03 / Model Evidence</p>
                          <h2 className="mt-1.5 text-lg font-semibold tracking-tight">Visual Explanation</h2>
                        </div>
                        <div className="flex flex-wrap justify-end gap-3 text-[11px] text-muted-foreground">
                          <span className="inline-flex items-center gap-1.5">
                            <i className="w-2 h-2 rounded-sm bg-[var(--accent-warn)]" /> Supports AI
                          </span>
                          <span className="inline-flex items-center gap-1.5">
                            <i className="w-2 h-2 rounded-sm bg-[var(--accent-good)]" /> Opposes AI
                          </span>
                        </div>
                      </div>

                      <div className="grid grid-cols-1 md:grid-cols-2 gap-3.5">
                        <figure className="m-0 overflow-hidden border border-border rounded-lg bg-secondary">
                          <figcaption className="px-3 py-2 border-b border-border font-mono text-[10px] uppercase tracking-wider text-muted-foreground">Original</figcaption>
                          {preview && <img src={preview} alt="Original image" className="block w-full aspect-[16/10] object-contain hover:scale-[1.012] transition-transform duration-200" />}
                        </figure>
                        <figure className="m-0 overflow-hidden border border-border rounded-lg bg-secondary">
                          <figcaption className="px-3 py-2 border-b border-border font-mono text-[10px] uppercase tracking-wider text-muted-foreground">Evidence Map</figcaption>
                          <img
                            src={prediction.explanation_image}
                            alt="SignalScope model evidence overlay"
                            className="block w-full aspect-[16/10] object-contain hover:scale-[1.012] transition-transform duration-200"
                          />
                        </figure>
                      </div>

                      <div className="flex flex-col md:flex-row md:items-start justify-between gap-6 pt-4 px-0.5">
                        <p className="max-w-[700px] m-0 text-muted-foreground text-xs leading-relaxed">
                          {prediction.explanation_text ||
                            "This evidence map shows the regions that most influenced the model's decision."}
                        </p>

                        {prediction.explanation_evidence && (
                          <dl className="flex gap-6 shrink-0 m-0">
                            <div>
                              <dt className="text-[10px] text-muted-foreground">Max Support</dt>
                              <dd className="mt-1 font-mono text-xs text-foreground">
                                {(
                                  prediction.explanation_evidence
                                    .max_positive_influence * 100
                                ).toFixed(1)}
                                %
                              </dd>
                            </div>
                            <div>
                              <dt className="text-[10px] text-muted-foreground">Max Opposing</dt>
                              <dd className="mt-1 font-mono text-xs text-foreground">
                                {(
                                  Math.abs(
                                    prediction.explanation_evidence
                                      .max_negative_influence
                                  ) * 100
                                ).toFixed(1)}
                                %
                              </dd>
                            </div>
                          </dl>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        <section
          className="evaluation-section mt-10 pt-8 border-t border-border"
          aria-labelledby="evaluation-title"
        >
          <div className="flex flex-col md:flex-row md:items-end justify-between gap-4 mb-4">
            <div>
              <p className="m-0 text-muted-foreground font-mono text-[10px] font-semibold tracking-[0.13em] uppercase">04 / Evaluation</p>
              <h2 id="evaluation-title" className="mt-1.5 text-xl font-semibold tracking-tight">Generalization snapshot</h2>
            </div>
            <Badge variant="outline" className="font-mono text-[9px] uppercase tracking-wider text-muted-foreground bg-transparent border-border rounded-full px-2.5 py-1">Internal · pseudo-unseen</Badge>
          </div>

          <Alert className="flex flex-col md:flex-row md:items-center justify-between gap-4 p-4 border border-border border-l-2 border-l-primary rounded-lg bg-card">
            <div>
              <p className="m-0 mb-1 font-mono text-[9px] uppercase tracking-wider text-muted-foreground">
                Held-out generator used for internal testing
              </p>
              <AlertTitle className="text-lg font-semibold m-0 tracking-tight">VQDM</AlertTitle>
            </div>
            <AlertDescription className="max-w-[620px] text-[11px] text-muted-foreground leading-relaxed m-0">
              These results come from SignalScope&apos;s internal
              pseudo-unseen split. They are not the official organizer
              held-out test score.
            </AlertDescription>
          </Alert>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-3">
            <Card className="border-border bg-secondary shadow-none rounded-lg p-4">
              <span className="block text-[9px] font-mono uppercase tracking-wider text-muted-foreground">ROC-AUC</span>
              <strong className="block mt-2 text-2xl font-medium tracking-tight text-foreground">0.728</strong>
              <small className="block mt-1.5 text-[10px] text-muted-foreground/80 leading-snug">Primary generalization metric</small>
            </Card>
            <Card className="border-border bg-card shadow-none rounded-lg p-4">
              <span className="block text-[9px] font-mono uppercase tracking-wider text-muted-foreground">Macro-F1</span>
              <strong className="block mt-2 text-2xl font-medium tracking-tight text-foreground">0.511</strong>
              <small className="block mt-1.5 text-[10px] text-muted-foreground/80 leading-snug">Frozen validation threshold</small>
            </Card>
            <Card className="border-border bg-card shadow-none rounded-lg p-4">
              <span className="block text-[9px] font-mono uppercase tracking-wider text-muted-foreground">Accuracy</span>
              <strong className="block mt-2 text-2xl font-medium tracking-tight text-foreground">57.4%</strong>
              <small className="block mt-1.5 text-[10px] text-muted-foreground/80 leading-snug">1,000-image pseudo-unseen set</small>
            </Card>
            <Card className="border-border bg-card shadow-none rounded-lg p-4">
              <span className="block text-[9px] font-mono uppercase tracking-wider text-muted-foreground">FPR</span>
              <strong className="block mt-2 text-2xl font-medium tracking-tight text-foreground">6.6%</strong>
              <small className="block mt-1.5 text-[10px] text-muted-foreground/80 leading-snug">Real images falsely flagged</small>
            </Card>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-[1.25fr_0.75fr] gap-3 mt-3">
            <Card className="border-border bg-card shadow-none rounded-lg p-4 md:p-5">
              <div className="flex items-start justify-between gap-4">
                <div>
                  <p className="m-0 text-muted-foreground font-mono text-[10px] font-semibold tracking-[0.13em] uppercase">Decision behavior</p>
                  <h3 className="mt-1.5 text-[15px] font-semibold tracking-tight">Confusion matrix</h3>
                </div>
                <span className="font-mono text-[9px] text-muted-foreground">threshold 64.69%</span>
              </div>

              <div
                className="confusion-matrix"
                role="table"
                aria-label="Confusion matrix for internal pseudo-unseen VQDM evaluation"
              >
                <div className="matrix-corner" role="columnheader">
                  &nbsp;
                </div>
                <div className="matrix-axis" role="columnheader">
                  Pred. real
                </div>
                <div className="matrix-axis" role="columnheader">
                  Pred. AI
                </div>

                <div className="matrix-axis matrix-side" role="rowheader">
                  Actual real
                </div>
                <div className="matrix-cell matrix-good">
                  <strong>467</strong>
                  <span>True negative</span>
                </div>
                <div className="matrix-cell matrix-warn">
                  <strong>33</strong>
                  <span>False positive</span>
                </div>

                <div className="matrix-axis matrix-side" role="rowheader">
                  Actual AI
                </div>
                <div className="matrix-cell matrix-soft">
                  <strong>393</strong>
                  <span>False negative</span>
                </div>
                <div className="matrix-cell matrix-good">
                  <strong>107</strong>
                  <span>True positive</span>
                </div>
              </div>
            </Card>

            <Card className="border-border bg-card shadow-none rounded-lg p-4 md:p-5">
              <div className="flex items-start justify-between gap-4">
                <div>
                  <p className="m-0 text-muted-foreground font-mono text-[10px] font-semibold tracking-[0.13em] uppercase">Evaluation protocol</p>
                  <h3 className="mt-1.5 text-[15px] font-semibold tracking-tight">Generalization-first split</h3>
                </div>
              </div>

              <dl className="mt-4 pt-1 border-t border-border">
                <div className="flex items-baseline justify-between gap-4 py-2.5 border-b border-border">
                  <dt className="text-[10px] text-muted-foreground">Training</dt>
                  <dd className="m-0 font-mono text-[11px] text-foreground">22,000 images</dd>
                </div>
                <div className="flex items-baseline justify-between gap-4 py-2.5 border-b border-border">
                  <dt className="text-[10px] text-muted-foreground">Validation</dt>
                  <dd className="m-0 font-mono text-[11px] text-foreground">16,000 images</dd>
                </div>
                <div className="flex items-baseline justify-between gap-4 py-2.5 border-b border-border">
                  <dt className="text-[10px] text-muted-foreground">Pseudo-unseen</dt>
                  <dd className="m-0 font-mono text-[11px] text-foreground">1,000 VQDM images</dd>
                </div>
                <div className="flex items-baseline justify-between gap-4 py-2.5 border-b border-border">
                  <dt className="text-[10px] text-muted-foreground">Calibration</dt>
                  <dd className="m-0 font-mono text-[11px] text-foreground">Validation only</dd>
                </div>
              </dl>

              <p className="mt-3 text-[11px] text-muted-foreground leading-relaxed">
                The unseen generator is kept outside model training and
                calibration so the internal score measures transfer to a
                generator family the model did not see during training.
              </p>
            </Card>
          </div>
        </section>

        <footer className="mt-8 pt-4 border-t border-border text-[10px] text-muted-foreground">
          SignalScope · Likelihood-based AI image detection
        </footer>
      </div>
    </main>
  );
}
