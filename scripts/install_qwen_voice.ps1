python -m pip install -U `
  transformers==4.57.6 `
  accelerate==1.12.0 `
  qwen-omni-utils `
  librosa `
  soundfile `
  sox `
  nagisa==0.2.11 `
  soynlp==0.0.493 `
  modelscope `
  torchaudio `
  einops `
  onnxruntime

if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}

python -m pip install --no-deps -U qwen-asr==0.0.6 qwen-tts==0.1.1

if (Get-Command winget -ErrorAction SilentlyContinue) {
  winget install --id ChrisBagwell.SoX --exact --accept-source-agreements --accept-package-agreements --silent
}
