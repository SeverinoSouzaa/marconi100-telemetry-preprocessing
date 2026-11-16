"""Script de download e extração oficial do Marconi100 Dataset 1 (Zenodo Record 7588815).

Baixa os pacotes mensais oficiais do Zenodo com suporte a retomada HTTP Range,
verificação de integridade e descompactação em formato Apache Parquet.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DADOS_DIR = ROOT / 'dados'
RAW_DIR = DADOS_DIR / 'raw'
EXTRACT_DIR = DADOS_DIR / 'marconi100'

# URL base oficial do registro no Zenodo (Record 7588815)
ZENODO_RECORD_URL = 'https://zenodo.org/records/7588815/files/'

# Inventário dos arquivos mensais e tamanhos nominais
ARQUIVOS_ZENODO = {
    '20-04.tar': {
        'url': f'{ZENODO_RECORD_URL}20-04.tar?download=1',
        'tamanho_bytes': 210769920,
        'periodo': 'Abril de 2020 (Mês representativo e compacto)',
    },
    '20-03.tar': {
        'url': f'{ZENODO_RECORD_URL}20-03.tar?download=1',
        'tamanho_bytes': 3328593920,
        'periodo': 'Março de 2020',
    },
}


def formatar_bytes(num_bytes: int) -> str:
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if abs(num_bytes) < 1024.0:
            return f'{num_bytes:3.1f} {unit}'
        num_bytes /= 1024.0
    return f'{num_bytes:.1f} PB'


def baixar_arquivo(url: str, destino: Path, tamanho_esperado: int | None = None) -> Path:
    destino.parent.mkdir(parents=True, exist_ok=True)
    arquivo_parcial = destino.with_suffix('.partial')

    tamanho_existente = arquivo_parcial.stat().st_size if arquivo_parcial.exists() else 0

    if destino.exists():
        if tamanho_esperado is None or destino.stat().st_size == tamanho_esperado:
            print(f'-> Arquivo já existe e está completo: {destino.name} ({formatar_bytes(destino.stat().st_size)})')
            return destino
        else:
            print(f'-> Tamanho incorreto detectado. Reiniciando download de {destino.name}...')
            destino.unlink()

    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    if tamanho_existente > 0:
        headers['Range'] = f'bytes={tamanho_existente}-'
        modo = 'ab'
        print(f'-> Retomando download de {destino.name} a partir de {formatar_bytes(tamanho_existente)}...')
    else:
        modo = 'wb'
        print(f'-> Iniciando download de {destino.name} ({formatar_bytes(tamanho_esperado or 0)})...')

    req = urllib.request.Request(url, headers=headers)
    inicio = time.time()
    baixado = tamanho_existente

    with urllib.request.urlopen(req, timeout=60) as resp, open(arquivo_parcial, modo) as out:
        total = int(resp.headers.get('Content-Length', 0)) + tamanho_existente
        bloco = 512 * 1024  # 512 KB

        while True:
            chunk = resp.read(bloco)
            if not chunk:
                break
            out.write(chunk)
            baixado += len(chunk)
            decorrido = time.time() - inicio
            vel = (baixado - tamanho_existente) / (decorrido + 1e-6)
            pct = (baixado / total) * 100 if total > 0 else 0
            sys.stdout.write(
                f'\r   Progresso: {pct:5.1f}% | {formatar_bytes(baixado)}/{formatar_bytes(total)} '
                f'| {formatar_bytes(int(vel))}/s'
            )
            sys.stdout.flush()

    print()
    arquivo_parcial.replace(destino)
    print(f'-> Download concluído com sucesso: {destino.name}')
    return destino


def extrair_tar(caminho_tar: Path, destino_extracao: Path) -> Path:
    destino_extracao.mkdir(parents=True, exist_ok=True)
    print(f'-> Descompactando {caminho_tar.name} em {destino_extracao}...')

    with tarfile.open(caminho_tar, 'r:*') as tar:
        tar.extractall(destino_extracao)

    print('-> Descompactação finalizada.')
    return destino_extracao


def executar_download(mes: str = '20-04') -> Path:
    nome_tar = f'{mes}.tar'
    if nome_tar not in ARQUIVOS_ZENODO:
        raise ValueError(f'Mês {mes} não configurado. Opções válidas: {list(ARQUIVOS_ZENODO.keys())}')

    info = ARQUIVOS_ZENODO[nome_tar]
    destino_tar = RAW_DIR / nome_tar
    baixar_arquivo(info['url'], destino_tar, info['tamanho_bytes'])

    pasta_mes = EXTRACT_DIR / mes
    if not pasta_mes.exists() or not any(pasta_mes.iterdir()):
        extrair_tar(destino_tar, pasta_mes)
    else:
        print(f'-> Dados já descompactados em {pasta_mes}')

    return pasta_mes


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download dos pacotes de telemetria do Marconi100 no Zenodo.')
    parser.add_argument('--mes', type=str, default='20-04', help='Mês para download (padrão: 20-04)')
    args = parser.parse_args()
    executar_download(args.mes)
