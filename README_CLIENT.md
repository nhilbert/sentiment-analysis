# Deutsches Kundenfeedback-Analysesystem

## 🎯 Überblick

Dieses produktionsreife System analysiert deutsches Kundenfeedback mittels fortschrittlicher semantischer Clusteranalyse und Sentimentanalyse. Das System passt sich automatisch an verfügbare Pakete an und bietet intelligente Fallback-Lösungen für maximale Funktionalität in jeder Umgebung.

## ✨ Hauptfunktionen

- **Semantische Clusteranalyse**: Nutzt neuronale Embeddings (SentenceTransformers) mit HDBSCAN für echtes semantisches Verständnis
- **Deutsch-Optimiert**: Spezialisierte deutsche Stoppwörter, Sentimentanalyse und Cluster-Beschriftung
- **Adaptive Technologie**: Erkennt automatisch verfügbare Pakete und bietet Fallback-Lösungen
- **Professionelle Visualisierungen**: Deutschsprachige Dashboards und Diagramme
- **Mehrere Ausgabeformate**: CSV, Excel und PNG-Visualisierungen
- **Produktionsbereit**: Umfassendes Logging, Fehlerbehandlung und Dokumentation

## 🚀 Schnellstart

### 1. Grundlegende Verwendung

```python
from sentiment_analysis import GermanFeedbackAnalyzer

# Analyzer initialisieren
analyzer = GermanFeedbackAnalyzer('data/ihre_feedback_datei.xlsx')

# Vollständige Analyse durchführen
results_df, summary_df = analyzer.run_analysis()
```

### 2. Kommandozeilen-Verwendung

```bash
python3 sentiment_analysis.py
```

## 📋 Systemanforderungen

### Mindestanforderungen (Grundfunktionalität)
- Python 3.7+
- pandas
- numpy

### Empfohlen für vollständige Funktionalität
```bash
pip install pandas numpy scikit-learn sentence-transformers umap-learn hdbscan vaderSentiment nltk matplotlib pyparsing seaborn openpyxl
```

### Installationsstufen

**Stufe 1 - Basis (Minimale Abhängigkeiten)**
```bash
pip install pandas numpy
```
- Grundlegende Worthäufigkeits-Clusteranalyse
- Regelbasierte deutsche Sentimentanalyse
- Nur CSV-Ausgabe

**Stufe 2 - Erweitert**
```bash
pip install pandas numpy scikit-learn nltk matplotlib pyparsing openpyxl
```
- TF-IDF-Clusteranalyse mit deutschen Stoppwörtern
- VADER-Sentimentanalyse
- Grundlegende Visualisierungen
- Excel-Unterstützung

**Stufe 3 - Professionell (Empfohlen)**
```bash
pip install pandas numpy scikit-learn sentence-transformers umap-learn hdbscan vaderSentiment nltk matplotlib pyparsing seaborn openpyxl
```
- Neuronale semantische Embeddings
- HDBSCAN-Clusteranalyse mit optimierten Parametern
- Professionelle deutsche Visualisierungen
- Vollständiger Funktionsumfang

## 📊 Eingabedatenformat

Ihre Eingabedatei sollte folgende Spalten enthalten:

| Spalte | Beschreibung | Erforderlich |
|--------|--------------|-------------|
| `date` | Feedback-Datum (YYYY-MM-DD) | Ja |
| `message` | Kundenfeedback-Text | Ja |

**Unterstützte alternative Spaltennamen:**
- `nachricht`, `message_de` → `message`
- `datum`, `created_at` → `date`

**Beispiel CSV:**
```csv
date,message
2024-01-15,"Der Kundensupport war sehr hilfreich und freundlich."
2024-01-16,"Die App-Oberfläche ist auf dem Handy schlecht."
2024-01-17,"Lieferung kam pünktlich an, vielen Dank!"
```

## 📈 Ausgabedateien

Das System generiert mehrere Ausgabedateien im `output/` Verzeichnis:

### Datendateien
- `feedback_analysis_results.csv` - Vollständige Analyse mit Clustern und Sentiment
- `cluster_summary.csv` - Statistische Zusammenfassung jedes Clusters
- `feedback_analysis_complete.xlsx` - Excel-Arbeitsmappe mit mehreren Arbeitsblättern

### Visualisierungen
- `semantic_clustering_dashboard.png` - Haupt-Dashboard mit 4 wichtigen Diagrammen
- `feedback_analysis.log` - Detailliertes Ausführungsprotokoll

## 🎨 Visualisierungs-Dashboard

Das System erstellt ein umfassendes deutsches Dashboard mit:

1. **Top 10 Semantische Cluster** - Balkendiagramm der häufigsten Themen
2. **Sentiment-Verteilung** - Kreisdiagramm mit deutschen Beschriftungen (Positiv/Neutral/Negativ)
3. **Sentiment über Zeit** - Zeitreihe der Sentiment-Trends
4. **Cluster-Sentiment Heatmap** - Sentiment-Verteilung über Cluster hinweg

## 🔧 Erweiterte Konfiguration

### Benutzerdefinierte Eingabedatei
```python
analyzer = GermanFeedbackAnalyzer('pfad/zu/ihren/daten.xlsx')
```

### Zugriff auf Ergebnisse
```python
results_df, summary_df = analyzer.run_analysis()

# Zugriff auf Cluster-Informationen
print(analyzer.cluster_meta)

# Top-Cluster abrufen
top_clusters = summary_df.head(10)
```

### Technologie-Erkennung
```python
# Verfügbare Funktionen prüfen
print(analyzer.capabilities)
# Ausgabe: {'sklearn': True, 'sentence_transformers': True, ...}
```

## 📊 Analyse-Pipeline

Das System folgt dieser Analyse-Pipeline:

1. **Datenladen & Bereinigung** - Excel/CSV laden, Spalten validieren, Text bereinigen
2. **Text-Embedding** - Semantische Embeddings erstellen (neuronal > TF-IDF > basis)
3. **Dimensionsreduktion** - UMAP für bessere Clusterbildung anwenden
4. **Semantische Clusteranalyse** - HDBSCAN mit optimierten Parametern
5. **Deutsche Beschriftung** - Aussagekräftige deutsche Cluster-Namen generieren
6. **Sentimentanalyse** - VADER oder regelbasierte deutsche Sentimentanalyse
7. **Visualisierung** - Deutsches Dashboard erstellen
8. **Ergebnisexport** - In mehreren Formaten speichern

## 🎯 Clusterqualität

Das System verwendet fein abgestimmte Parameter für optimale semantische Trennung:

- **Neuronale Embeddings**: `min_cluster_size=5, min_samples=2, cluster_selection_epsilon=0.1`
- **TF-IDF Embeddings**: `min_cluster_size=30, min_samples=10`
- **Deutscher Kontext**: Spezialisierte Begriffszuordnung und kontextuelle Beschriftung

## 🌍 Deutsche Sprachunterstützung

### Sentimentanalyse
- Umfassende deutsche Positiv-/Negativ-Wortlisten
- Geschäftsbereich-Terminologie
- Unterstützung für gemischte deutsch/englische Inhalte

### Cluster-Beschriftung
- Kontextuelle deutsche Geschäftsbegriffe
- Bereichsspezifische Übersetzungen
- Aussagekräftige semantische Gruppierungen

### Visualisierungen
- Deutsche Diagrammtitel und Beschriftungen
- Korrekte deutsche Formatierung
- Geschäftsangemessene Gestaltung

## 🔍 Fehlerbehebung

### Häufige Probleme

**"Erforderliche Spalten fehlen"**
- Stellen Sie sicher, dass Ihre Datei `date` und `message` Spalten hat
- Prüfen Sie alternative Spaltennamen (siehe Eingabedatenformat)

**"Keine gültigen Daten nach Bereinigung"**
- Prüfen Sie das Datumsformat (sollte von pandas parsbar sein)
- Stellen Sie sicher, dass die Nachrichtenspalte Text enthält (nicht leer/null)

**"Paket nicht verfügbar"**
- Installieren Sie empfohlene Pakete für vollständige Funktionalität
- Das System verwendet automatisch Fallback-Lösungen

### Leistungstipps

- Für große Datensätze (>10.000 Nachrichten) stellen Sie ausreichend RAM sicher
- Neuronale Embeddings benötigen mehr Speicher, liefern aber bessere Ergebnisse
- Verwenden Sie SSD-Speicher für schnellere Datei-E/A

## 📞 Support

### Protokolldateien
Prüfen Sie `feedback_analysis.log` für detaillierte Ausführungsinformationen.

### Funktionserkennung
Das System protokolliert, welche Pakete verfügbar sind:
```
INFO: ✓ sentence_transformers verfügbar
INFO: ✓ hdbscan verfügbar
INFO: ✗ matplotlib nicht verfügbar - verwende Fallback
```

### Fehlerbehandlung
Alle Fehler werden mit Zeitstempel und detaillierten Nachrichten für die Fehlersuche protokolliert.

## 🚀 Produktionsbereitstellung

### Umgebungseinrichtung
1. Python 3.7+ und erforderliche Pakete installieren
2. Eingabedateien in `data/` Verzeichnis platzieren
3. Sicherstellen, dass `output/` Verzeichnis beschreibbar ist
4. Analyseskript ausführen

### Automatisierung
Das System kann einfach in automatisierte Arbeitsabläufe integriert werden:
```bash
# Cron-Job Beispiel
0 9 * * 1 cd /pfad/zur/analyse && python3 sentiment_analysis.py
```

### Überwachung
- Protokolldateien auf Ausführungsstatus prüfen
- Ausgabeverzeichnis auf Ergebnisse überwachen
- Cluster-Anzahl und Sentiment-Verteilung validieren

## 📋 Versionshistorie

- **v1.0** - Erste Produktionsversion mit vollständiger deutscher Unterstützung
- Fortschrittliche semantische Clusteranalyse mit neuronalen Embeddings
- Umfassende deutsche Sentimentanalyse
- Professionelles Visualisierungs-Dashboard
- Produktionsbereite Fehlerbehandlung und Protokollierung

---

**System entwickelt für professionelle deutsche Kundenfeedback-Analyse mit Unternehmensqualität in Zuverlässigkeit und Leistung.**