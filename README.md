# Sentiment-Analyse
Ein Python-basiertes System zur semantischen Analyse von deutschem Kundenfeedback

## Überblick

Dieses System analysiert deutsches Kundenfeedback automatisch und identifiziert:

### Was wird analysiert?
- **Semantische Cluster**: Automatische Gruppierung ähnlicher Themen und Beschwerden
- **Sentiment-Bewertung**: Positive, neutrale und negative Stimmung pro Nachricht
- **Zeitliche Trends**: Entwicklung der Kundenstimmung über Zeit
- **Häufige Themen**: Die wichtigsten Anliegen und Feedback-Kategorien

### Wie funktioniert es?
1. **Text-Embeddings**: Umwandlung der Texte in numerische Vektoren mit SentenceTransformers
2. **Clustering**: Gruppierung ähnlicher Nachrichten mit HDBSCAN-Algorithmus
3. **Sentiment-Analyse**: Bewertung der Stimmung mit German BERT (optimiert für Deutsch)
4. **Visualisierung**: Automatische Erstellung von Diagrammen und Dashboards

### Was können Sie erwarten?
- **Cluster-Labels**: Deutsche Bezeichnungen für entdeckte Themen (z.B. "Support Rückruf", "Anmelde-Probleme")
- **Sentiment-Scores**: Numerische Bewertung von -1 (sehr negativ) bis +1 (sehr positiv)
- **Statistiken**: Anzahl Nachrichten pro Thema, durchschnittliche Stimmung, zeitliche Verteilung
- **Dashboard**: Übersichtliche Visualisierung aller Ergebnisse

## Installation

### 1. Virtuelle Umgebung erstellen
```bash
python3 -m venv venv
```

### 2. Virtuelle Umgebung aktivieren
**Linux/macOS:**
```bash
source venv/bin/activate
```

**Windows:**
```bash
venv\Scripts\activate
```

### 3. Abhängigkeiten installieren
```bash
pip install -r requirements.txt
```

## Ausführung

### Analyse starten
```bash
python main.py
```

### Tests ausführen
```bash
python run_tests.py
```

### Entwicklungsumgebung einrichten
```bash
pip install -r requirements-dev.txt
```

### Datenformat
Die Eingabedatei sollte zwei Spalten enthalten:
- `date`: Datum des Feedbacks
- `feedback`: Deutscher Feedback-Text

### Ergebnisse
Die Analyse erstellt folgende Ausgabedateien im `output/` Ordner:
- `feedback_analysis_results.csv`: Vollständige Analyseergebnisse
- `cluster_summary.csv`: Cluster-Zusammenfassung
- `feedback_analysis_complete.xlsx`: Excel-Datei (falls verfügbar)
- `semantic_clustering_dashboard.png`: Visualisierung

### Virtuelle Umgebung deaktivieren
```bash
deactivate
```

## Beispiel-Ergebnisse

Nach der Analyse erhalten Sie:

**Sentiment-Verteilung:**
- 😊 Positive Nachrichten: 32%
- 😐 Neutrale Nachrichten: 28%
- 😞 Negative Nachrichten: 40%

**Top Themen:**
1. Anmelde-Probleme (29 Nachrichten, -0.763 Sentiment)
2. Versand & Lieferung (21 Nachrichten, 0.010 Sentiment)
3. Benutzeroberfläche (18 Nachrichten, 0.905 Sentiment)

## Technische Anforderungen

**Minimal:**
- Python 3.8+
- 4GB RAM
- 1GB freier Speicherplatz

**Empfohlen:**
- Python 3.9+
- 8GB RAM
- GPU für schnellere Verarbeitung (optional)

## Support

Bei Fragen oder Problemen:
1. Prüfen Sie die Log-Datei `feedback_analysis.log`
2. Stellen Sie sicher, dass alle Abhängigkeiten installiert sind
3. Überprüfen Sie das Datenformat (date + feedback Spalten)

Das System erkennt automatisch verfügbare Pakete und passt sich entsprechend an - es funktioniert auch mit minimaler Installation.