# WallSpace AI inference

Mały serwis HTTP (FastAPI + onnxruntime) do segmentacji ścian (SegFormer / ADE20K),
działający na Google Cloud Run. Docelowo wspólne „AI-zaplecze": kolejne modele = kolejne
endpointy, wołane zarówno przez sklep (Next), jak i przez n8n.

Endpointy:
- `GET /health` → `{"ok": true}`
- `POST /segment` (multipart, pole `image`) → PNG maski (biel = ściana), w rozmiarze wgranego zdjęcia
  - wymaga nagłówka `X-API-Key: <sekret>`, jeśli ustawiono zmienną `API_SECRET`

---

## 1. Wrzuć kod na GitHub

```bash
cd wallspace-ai-inference
git init && git add . && git commit -m "AI inference service"
# utwórz puste repo na GitHub, potem:
git remote add origin git@github.com:<ty>/wallspace-ai-inference.git
git push -u origin main
```

## 2. Deploy na Cloud Run (z konsoli, bez lokalnego Dockera)

1. [console.cloud.google.com](https://console.cloud.google.com) → wybierz/utwórz projekt.
2. W wyszukiwarce u góry wpisz **Cloud Run** → wejdź.
3. **Create service** → **Continuously deploy from a repository** → **Set up with Cloud Build**.
4. Połącz konto GitHub → wybierz repo `wallspace-ai-inference`, branch `main`.
5. Build type: **Dockerfile** (zostaw domyślną ścieżkę `/Dockerfile`). **Save**.
6. Ustawienia usługi:
   - **Authentication**: *Allow unauthenticated invocations* (chronimy sekretem w nagłówku).
   - **Region**: `europe-central2` (Warszawa) lub `europe-west1`.
   - **Resources**: Memory **2 GiB**, CPU **2**.
   - **Requests**: Request timeout **120**, Max concurrent requests per instance **4**.
   - **Autoscaling**: Min instances **0**, Max instances **5**.
   - Rozwiń **Container, Networking, Security** → włącz **Startup CPU boost** (szybszy zimny start).
   - **Variables & Secrets** → dodaj zmienną `API_SECRET` = długi losowy ciąg (zapisz go — wkleisz w sklepie i w n8n).
7. **Create**. Po chwili dostaniesz URL: `https://wallspace-ai-inference-xxxx.run.app`.

> Od teraz każdy `git push` na `main` automatycznie przebuduje i wdroży usługę.

## 3. Test

```bash
curl https://<twoj-url>.run.app/health
# {"ok":true}

curl -X POST https://<twoj-url>.run.app/segment \
  -H "X-API-Key: <sekret>" \
  -F "image=@pokoj.jpg" \
  --output maska.png
# maska.png — biel tam, gdzie model widzi ścianę
```

Pierwsze wywołanie po przerwie (min=0) potrwa ~4–10 s (zimny start). Kolejne ~1–3 s.

## 4. Użycie z n8n

Node **HTTP Request**: metoda `POST`, URL `https://<url>.run.app/segment`,
nagłówek `X-API-Key: <sekret>`, body typu `Form-Data` z polem `image` (binarne).

## Konfiguracja (zmienne środowiskowe)

| Zmienna | Domyślnie | Opis |
|---|---|---|
| `API_SECRET` | — | jeśli ustawione, każdy request musi mieć nagłówek `X-API-Key` |
| `ORT_THREADS` | `2` | liczba wątków CPU (zrównaj z CPU usługi) |
| `MODEL_PATH` | `model.onnx` | ścieżka do modelu (zapieczony w obrazie) |

## Zmiana modelu / jakości

Domyślnie pełna precyzja `b2` (najlepsza jakość). Mniejszy/szybszy obraz:
```
--build-arg MODEL_FILE=model_quantized.onnx     # ~29 MB zamiast ~110 MB
--build-arg MODEL_REPO=Xenova/segformer-b4-finetuned-ade-512-512   # większy model = lepsza jakość
```

## Test lokalny (opcjonalnie)

```bash
pip install -r requirements.txt
python -c "import urllib.request as u; u.urlretrieve('https://huggingface.co/Xenova/segformer-b2-finetuned-ade-512-512/resolve/main/onnx/model.onnx','model.onnx')"
uvicorn main:app --reload --port 8080
```
