#!/bin/bash

AMBIENTE="base"
APP="facecc-ia"
PORT=7011

RUTA_APP="$HOME/facecc/$APP"
RUTA_SERVER="$HOME/facecc/apache-$APP"
RUTA_LOGS="$HOME/facecc/logs"

source "$HOME/miniforge3/bin/activate" "$AMBIENTE" && \
cd "$RUTA_APP" && \
mkdir -p "$RUTA_LOGS" && \
mod_wsgi-express start-server application.wsgi --port $PORT \
      --server-root "$RUTA_SERVER" \
      --access-log --log-to-terminal \
       2>&1 | /usr/bin/cronolog "$RUTA_LOGS/$APP.%Y-%m-%d.log"
