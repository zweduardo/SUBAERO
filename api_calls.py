# api_calls.py — Chamadas às APIs (VRA / Clima / OpenSky)
import requests
import json
import time


# ── Retry helper ────────────────────────────────────────────────────────────
DEFAULT_TIMEOUT = 30          # segundos por request
MAX_RETRIES     = 3
BACKOFF_BASE    = 5           # espera 5s, 10s, 20s …

def _get_json(url, timeout=DEFAULT_TIMEOUT, retries=MAX_RETRIES):
    """GET com retry + backoff exponencial em caso de timeout/erro de rede."""
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except (requests.exceptions.Timeout,
                requests.exceptions.ConnectionError) as exc:
            last_exc = exc
            wait = BACKOFF_BASE * (2 ** (attempt - 1))
            print(f"  [retry {attempt}/{retries}] {type(exc).__name__} — "
                  f"tentando novamente em {wait}s…")
            time.sleep(wait)
        except requests.exceptions.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 429:
                wait = BACKOFF_BASE * (2 ** (attempt - 1))
                print(f"  [retry {attempt}/{retries}] Rate-limited (429) — "
                      f"tentando novamente em {wait}s…")
                time.sleep(wait)
                last_exc = exc
            else:
                raise
    raise last_exc  # esgotou retries


# ── VRA (ANAC) ──────────────────────────────────────────────────────────────
def get_vra_data(date):
    """Retorna voos de uma data (formato ddMMyyyy)."""
    url = f"https://sas.anac.gov.br/sas/vra_api/vra/data?dt_voo={date}"
    return _get_json(url, timeout=60)

def get_vra_period_data(start_date, end_date):
    """Retorna voos de um período."""
    url = (
        f"https://sas.anac.gov.br/sas/vra_api/vra"
        f"?dt_referencia1={start_date}&dt_referencia2={end_date}"
    )
    return _get_json(url, timeout=120)

def get_vra_aerodromo_data(aerodrome_code):
    """Retorna dados de um aeródromo pelo código ICAO ou IATA."""
    url = (
        f"https://sas.anac.gov.br/sas/vra_api/aerodromo"
        f"?sg_aerodromo_icao_ou_iata={aerodrome_code}"
    )
    return _get_json(url)

def get_vra_airline_data(date, airline_code, origin_code, destination_code, flight_number):
    """Retorna dados de um voo específico."""
    url = (
        f"https://sas.anac.gov.br/sas/vra_api/vra/voo"
        f"?dt_voo={date}"
        f"&sg_empresa_icao={airline_code}"
        f"&sg_icao_origem={origin_code}"
        f"&sg_icao_destino={destination_code}"
        f"&nr_voo={flight_number}"
    )
    return _get_json(url)

def get_siros_temp(temporada):
    """Retorna dados SIROS de uma temporada."""
    url = f"https://sas.anac.gov.br/sas/siros_api/ssimfile?ds_temporada={temporada}"
    return _get_json(url)


# ── Clima (OpenWeatherMap) ──────────────────────────────────────────────────
import os
API_KEY_CLIMA = os.getenv("API_KEY_CLIMA", "your_openweathermap_api_key_here")

def get_clima(lat, lon):
    """Retorna clima atual para uma coordenada."""
    url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?lat={lat}&lon={lon}&appid={API_KEY_CLIMA}"
    )
    return _get_json(url)

def get_clima_period(start_date, end_date, lat, lon):
    """Retorna clima histórico para uma coordenada."""
    url = (
        f"https://api.openweathermap.org/data/2.5/onecall/timemachine"
        f"?lat={lat}&lon={lon}&dt={start_date}&appid={API_KEY_CLIMA}"
    )
    return _get_json(url)

def get_clima_historico(lat, lon, unix_timestamp):
    """Retorna clima histórico para uma coordenada num timestamp Unix específico."""
    url = (
        f"https://api.openweathermap.org/data/2.5/onecall/timemachine"
        f"?lat={lat}&lon={lon}&dt={unix_timestamp}&appid={API_KEY_CLIMA}"
    )
    data = _get_json(url)
    # A API timemachine retorna lista em "data"; pegar o registro mais próximo
    if "data" in data and data["data"]:
        return data["data"][0]
    # Fallback: retorna o dict inteiro (pode ter "current")
    return data

def city_name_to_coordinates(city_name):
    """Converte nome de cidade em coordenadas (lat, lon)."""
    url = (
        f"http://api.openweathermap.org/geo/1.0/direct"
        f"?q={city_name}&limit=1&appid={API_KEY_CLIMA}"
    )
    data = _get_json(url)
    if data:
        return data[0]["lat"], data[0]["lon"]
    return None, None

def tratar_clima(clima):
    """Extrai campos relevantes do JSON de clima e retorna dict plano."""
    weather_list = clima.get("weather", [])
    if not weather_list:
        return None
    weather = weather_list[0]
    main = clima.get("main", {})
    return {
        "weather":     weather.get("main"),
        "description": weather.get("description"),
        "temp":        main.get("temp"),
        "windspeed":   clima.get("wind", {}).get("speed"),
        "rain":        clima.get("rain", {}).get("1h", 0),
        "clouds":      clima.get("clouds", {}).get("all", 0),
    }


# ── Open-Meteo (gratuito, sem chave) ────────────────────────────────────────
# WMO Weather Code → descrição curta
_WMO_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Rime fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
    80: "Slight showers", 81: "Moderate showers", 82: "Violent showers",
    95: "Thunderstorm", 96: "Thunderstorm + hail", 99: "Thunderstorm + heavy hail",
}

def _wmo_to_weather(code):
    """Converte WMO weather code em (weather, description)."""
    if code is None:
        return "Unknown", "unknown"
    desc = _WMO_CODES.get(code, f"WMO {code}")
    if code <= 3:
        cat = "Clear" if code <= 1 else "Clouds"
    elif code <= 48:
        cat = "Fog"
    elif code <= 55:
        cat = "Drizzle"
    elif code <= 65:
        cat = "Rain"
    elif code <= 75:
        cat = "Snow"
    elif code <= 82:
        cat = "Rain"
    else:
        cat = "Thunderstorm"
    return cat, desc



# ── OpenSky Network ──────────────────────────────────────────────────────────
# Autenticação OAuth2 (client_credentials).
# Carrega credenciais de credentials.json se existir.

import os

_OPENSKY_BASE       = "https://opensky-network.org/api"
_OPENSKY_TOKEN_URL  = (
    "https://auth.opensky-network.org/auth/realms/opensky-network"
    "/protocol/openid-connect/token"
)
_opensky_token:      str | None = None
_opensky_token_exp:  float      = 0.0   # unix timestamp de expiração
_opensky_client_id:  str | None = None
_opensky_client_sec: str | None = None


def load_opensky_credentials(path: str = "credentials.json") -> bool:
    """
    Carrega clientId / clientSecret do arquivo JSON.
    Retorna True se carregou com sucesso.
    """
    global _opensky_client_id, _opensky_client_sec
    try:
        with open(path, encoding="utf-8") as f:
            creds = json.load(f)
        _opensky_client_id  = creds["clientId"]
        _opensky_client_sec = creds["clientSecret"]
        return True
    except Exception as exc:
        print(f"  [OpenSky] Não foi possível carregar {path}: {exc}")
        return False


def _opensky_get_token() -> str | None:
    """
    Obtém (ou renova) o Bearer token via client_credentials.
    Retorna o token string ou None em caso de falha.
    """
    global _opensky_token, _opensky_token_exp
    now = time.time()
    if _opensky_token and now < _opensky_token_exp - 30:
        return _opensky_token
    if not _opensky_client_id or not _opensky_client_sec:
        return None
    try:
        resp = requests.post(
            _OPENSKY_TOKEN_URL,
            data={
                "grant_type":    "client_credentials",
                "client_id":     _opensky_client_id,
                "client_secret": _opensky_client_sec,
            },
            timeout=15,
        )
        resp.raise_for_status()
        body = resp.json()
        _opensky_token     = body["access_token"]
        _opensky_token_exp = now + body.get("expires_in", 300)
        return _opensky_token
    except Exception as exc:
        print(f"  [OpenSky] Falha ao obter token: {exc}")
        return None


def _opensky_get(path: str, params: dict | None = None):
    """
    GET autenticado para OpenSky REST API com retry.
    Retorna o JSON parseado ou None em caso de erro.
    """
    token = _opensky_get_token()
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    url = _OPENSKY_BASE + path

    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, headers=headers,
                                timeout=DEFAULT_TIMEOUT)
            if resp.status_code == 404:
                return None
            if resp.status_code == 429:
                wait = BACKOFF_BASE * (2 ** (attempt - 1))
                print(f"  [OpenSky] Rate-limited — aguardando {wait}s…")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except (requests.exceptions.Timeout,
                requests.exceptions.ConnectionError) as exc:
            last_exc = exc
            wait = BACKOFF_BASE * (2 ** (attempt - 1))
            print(f"  [OpenSky] {type(exc).__name__} (tentativa {attempt}) — "
                  f"aguardando {wait}s…")
            time.sleep(wait)
    return None


def get_opensky_flights_by_time(begin_unix: int, end_unix: int) -> list[dict]:
    """
    Retorna todos os voos num intervalo de tempo (máx 2h por chamada).

    Cada item retornado tem:
      icao24              — hex ICAO24 da aeronave
      callsign            — callsign (ex: 'GLO1234  '), strip() para limpar
      estDepartureAirport — ICAO do aeroporto de origem estimado
      estArrivalAirport   — ICAO do aeroporto de destino estimado
      firstSeen / lastSeen — unix timestamps
    """
    data = _opensky_get("/flights/all",
                        params={"begin": begin_unix, "end": end_unix})
    if not data:
        return []
    return data


def get_opensky_aircraft_metadata(icao24: str) -> dict | None:
    """
    Retorna metadados de uma aeronave pelo hex ICAO24.

    Campos relevantes:
      registration  — matrícula (ex: 'PR-GXI')
      manufacturerName
      typecode      — ICAO type (ex: 'B738')
      built         — ano de fabricação (int ou str)
      operator
    Retorna None se a aeronave não for encontrada.
    """
    return _opensky_get(f"/metadata/aircraft/icao/{icao24.lower()}")


def get_opensky_aircraft_age(icao24: str) -> float | None:
    """
    Retorna a idade em anos da aeronave identificada pelo hex ICAO24.
    Retorna None se não houver dado de ano de fabricação.
    """
    meta = get_opensky_aircraft_metadata(icao24)
    if not meta:
        return None
    built = meta.get("built")
    if not built:
        return None
    try:
        year = int(str(built)[:4])
        from datetime import datetime
        return float(datetime.now().year - year)
    except (ValueError, TypeError):
        return None


def get_opensky_callsign_to_icao24(
    callsign: str,
    begin_unix: int,
    end_unix: int,
) -> str | None:
    """
    Dado um callsign e uma janela de tempo, retorna o ICAO24 hex
    da aeronave que operou esse voo.
    Retorna None se não encontrar.
    """
    flights = get_opensky_flights_by_time(begin_unix, end_unix)
    target = callsign.strip().upper()
    for f in flights:
        cs = (f.get("callsign") or "").strip().upper()
        if cs == target:
            return f.get("icao24")
    return None


def get_opensky_def():
    """Retorna True se as credenciais OpenSky estão carregadas."""
    return bool(_opensky_client_id and _opensky_client_sec)


def get_clima_openmeteo(lat, lon, start_date, end_date):
    """
    Busca clima histórico horário via Open-Meteo (gratuito).
    start_date / end_date: 'YYYY-MM-DD'
    Retorna lista de dicts, um por hora.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,windspeed_10m,rain,cloudcover,weathercode",
        "timezone": "UTC",
    }
    import requests
    for attempt in range(1, 6):
        resp = requests.get(url, params=params, timeout=60)
        if resp.status_code == 429:
            wait = 30 * attempt  # 30s, 60s, 90s, 120s, 150s
            print(f"    [Open-Meteo] Rate limit (tentativa {attempt}) - aguardando {wait}s...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        break
    data = resp.json()
    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    results = []
    for i, ts in enumerate(times):
        # ts format: '2025-01-01T14:00'
        date_part = ts[:10]          # 'YYYY-MM-DD'
        hour_part = ts[11:13]        # 'HH'
        wmo = hourly["weathercode"][i]
        cat, desc = _wmo_to_weather(wmo)
        results.append({
            "date_iso": date_part,
            "hour": hour_part,
            "weather": cat,
            "description": desc,
            "temp": hourly["temperature_2m"][i],
            "windspeed": hourly["windspeed_10m"][i],
            "rain": hourly["rain"][i] or 0,
            "clouds": hourly["cloudcover"][i] or 0,
        })
    return results
