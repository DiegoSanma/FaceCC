ACCION="$1"

if [[ "$ACCION" != "web" && "$ACCION" != "servicios" ]]; then
        echo "Error: Accion desconocida $ACCION"
        echo "Acciones validas: web servicios"
        exit 1
fi

if [[ "$ACCION" == "web" ]]; then
    APPS_DESTINATION="$HOME/facecc"
    runTest mkdir -p "$APPS_DESTINATION"
    for app in facecc facecc-ia 
    do
        echo "instalando $app en $APPS_DESTINATION ..."
        if [[ -f "$APPS_DESTINATION/bajar_$app.sh" ]]; then
            runTest bash "$APPS_DESTINATION/bajar_$app.sh"
        fi
        runTest cp -r "../$app/" "$APPS_DESTINATION/"
        runTest cp "subir_$app.sh" "bajar_$app.sh" "$APPS_DESTINATION"
        runTest chmod +x "$APPS_DESTINATION/subir_$app.sh" "$APPS_DESTINATION/bajar_$app.sh"
        runTest bash "$APPS_DESTINATION/subir_$app.sh"
    done
    echo "ok $ACCION"
elif [[ "$ACCION" == "servicios" ]]; then
    SERVICES_DIR="$HOME/.config/systemd/user/"
    runTest mkdir -p "$SERVICES_DIR"
    runTest cp server_facecc-front.service server_facecc-ia.service "$SERVICES_DIR"
	# configurar servicios para que suban en bootear
    runTest loginctl enable-linger
    runTest systemctl --user daemon-reload
    runTest systemctl --user enable server_facecc-front
    runTest systemctl --user enable server_facecc-ia
	# subir servicios
    runTest systemctl --user start server_facecc-front
    runTest systemctl --user start server_facecc-ia
    echo "ok $ACCION"
fi