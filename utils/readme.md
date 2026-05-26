# Google Drive Sync Launcher

`utils/launch.py`는 현재 리포를 Google Drive의 `MyDrive` 하위 폴더로 직접 동기화하는 로컬 전용 스크립트입니다.

의도는 다음과 같습니다.

- 로컬에서 코드 수정과 버전관리를 진행
- 필요한 시점에만 Google Drive로 현재 리포 상태를 미러링
- Colab에서는 Drive에 올라간 코드만 실행

즉, "개발은 로컬에서, 실행은 Colab에서"를 위한 업로더입니다.

## What It Does

스크립트는 아래 동작을 수행합니다.

- 리포 루트를 자동으로 찾음
- `.gitignore`에 걸리는 파일과 폴더를 제외함
- `.gitignore`, `.dockerignore`, `.ignore`, `.rgignore` 파일 자체도 제외함
- Google Drive의 대상 폴더를 생성하거나 재사용함
- 원격 폴더에서 ignore 대상이 아닌 기존 파일을 먼저 삭제함
- 현재 로컬 파일을 새로 업로드해서 실행용 상태를 다시 만듦
- 삭제와 업로드는 `tqdm` 진행 표시로 확인 가능

주의: 대상 Drive 폴더는 이 리포 전용 폴더로 쓰는 것이 좋습니다. `.gitignore`에 걸리지 않는 수동 파일은 삭제 대상이 될 수 있습니다.

## Install

로컬 환경에서만 아래 의존성이 필요합니다.

```bash
pip install -r requirements.launch.txt
```

`requirements.txt`는 Colab 실행용이고, `requirements.launch.txt`는 업로더용입니다.

## OAuth Setup

이 스크립트는 Google Drive API + OAuth 로그인 방식을 사용합니다.

필요한 것은 두 가지입니다.

- `client_secret.json`
- 로컬에 저장될 `token.json`

역할은 아래와 같습니다.

- `client_secret.json`: 처음 로그인할 때만 사용하는 OAuth 앱 설정 파일
- `token.json`: 한 번 로그인한 뒤 재사용하는 인증 토큰 파일

기본 토큰 저장 위치는 다음입니다.

```text
./utils/google_drive_token.json
```

첫 실행에서 브라우저 로그인이 열리고, 이후에는 저장된 토큰을 자동 재사용합니다.

## Basic Usage

예시:

```bash
python utils/launch.py \
  --dest-path Lab/GUARD_EXP_REPO \
  --client-secret ./utils/client_secret.json 
```

위 명령은 아래 Drive 경로를 기준으로 동작합니다.

```text
MyDrive/Lab/GUARD_EXP_REPO
```

`--dest-path`에는 `MyDrive`를 포함해도 되고 생략해도 됩니다.

예:

```bash
python utils/launch.py \
  --dest-path MyDrive/Lab/GUARD_EXP_REPO \
  --client-secret ./utils/client_secret.json \
```

## Useful Options

### Dry Run

실제 업로드 없이 변경 예정만 보고 싶다면:

```bash
python utils/launch.py \
  --dest-path Lab/GUARD_EXP_REPO \
  --client-secret ./utils/client_secret.json \
  --dry-run
```

### Verbose Output

파일 단위로 업로드/삭제/보존 내역까지 보고 싶다면:

```bash
python utils/launch.py \
  --dest-path Lab/GUARD_EXP_REPO \
  --client-secret ./utils/client_secret.json \
  --dry-run \
  --verbose
```

### Custom Token Path

토큰 파일 경로를 직접 지정하고 싶다면:

```bash
python utils/launch.py \
  --dest-path Lab/GUARD_EXP_REPO \
  --client-secret ./utils/client_secret.json \
  --token-path /path/to/google_drive_token.json
```

### Custom Repo Root

기본값은 현재 프로젝트 루트이지만, 다른 루트를 지정할 수도 있습니다.

```bash
python utils/launch.py \
  --repo-root /path/to/repo \
  --dest-path Lab/GUARD_EXP_REPO \
  --client-secret /utils/client_secret.json 
```

## Ignore Rules

업로드 대상에서 자동 제외되는 항목은 아래와 같습니다.

- 루트 `.gitignore`에 매칭되는 모든 파일/폴더
- `.git/`
- `__pycache__/`
- `*.pyc`
- `.gitignore`
- `.dockerignore`
- `.ignore`
- `.rgignore`

현재 프로젝트 기준으로는 보통 아래 항목이 제외됩니다.

- `/data`
- `/outputs`
- `/model/repo`
- `.venv/`

즉, 로컬 실험 산출물이나 대용량 모델 캐시는 Drive 동기화 대상에 포함되지 않습니다.

## Expected Output

실행하면 대략 아래 형태의 요약이 출력됩니다.

```text
Drive Path : MyDrive/Lab/GUARD_EXP_REPO
Ignored    : 12
Preserve   : 2
Upload     : 25
Delete     : 23
Result     : sync completed
```

`--dry-run`을 붙이면 실제 변경 없이 계획만 출력합니다.

## Colab Side

Colab에서는 보통 아래처럼 사용하면 됩니다.

```python
from google.colab import drive
drive.mount("/content/drive")
```

그 다음 동기화된 경로로 이동합니다.

```bash
cd /content/drive/MyDrive/Lab/GUARD_EXP_REPO
```

이제 Colab은 Drive에 올라간 최신 코드 기준으로 실행하면 됩니다.

## Recommended Workflow

권장 사용 흐름은 아래와 같습니다.

1. 로컬에서 코드 수정
2. 필요하면 테스트 실행
3. `--dry-run`으로 변경 예정 확인
4. 실제 동기화 실행
5. Colab에서 Drive 경로 기준으로 실행

## Security Notes

- `client_secret.json`은 리포에 커밋하지 않는 것을 권장합니다.
- 토큰 파일은 더 민감하므로 절대 커밋하면 안 됩니다.
- 계정을 바꾸거나 토큰이 꼬였으면 토큰 파일을 지우고 다시 로그인하면 됩니다.

## Quick Start

처음 세팅할 때는 아래 순서만 기억하면 충분합니다.

```bash
pip install -r requirements.launch.txt
python utils/launch.py \
  --dest-path Lab/GUARD_EXP_REPO \
  --client-secret ./utils/client_secret.json \
  --dry-run --verbose
python utils/launch.py \
  --dest-path Lab/GUARD_EXP_REPO \
  --client-secret ./utils/client_secret.json \
```
