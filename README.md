# Simulació i estudi analític d'un sistema de cues amb servei per lots: el cas d'una parada d'autobús urbà

Aquest repositori conté els annexos digitals i el material complementari del Treball de Final de Grau (TFG). Aquí es recullen els codis desenvolupats, els models de validació i els resultats obtinguts durant la investigació per a la seva consulta pública i avaluació.

## Contingut del Repositori

El material està organitzat en funció de les diferents fases de l'estudi:

### 1. Codi Principal (`Python`)
El projecte compta amb diferents scripts desenvolupats en **Python**:
* **Simulador de la parada de bus (`.py`):** Codi central que executa la simulació d'esdeveniments discrets del sistema de cues amb servei per lots.
* **Validació dels generadors (`.py`):** Script encarregat de verificar que els generadors de variables aleatòries utilitzats segueixen les distribucions teòriques de forma correcta.

### 2. Validació i Anàlisi del Simulador
* **Validació del simulador (`.py`):** Codi que contrasta els resultats obtinguts pel simulador amb la resolució analítica del model teòric de referència $M/M^k/1$.
* **Aproximació de Powell (`.xlsx`):** Full de càlcul d'Excel utilitzat com a mètode d'optimització i validació del simulador a través de l'aproximació de Powell.
* **Imatges i Taules de Resultats:** Gràfics i captures de les taules que recullen el comportament del sistema sota diferents escenaris i configuracions de càrrega.
* **Resultats Complementaris (`.txt`):** Fitxer de text que recopila manualment les sortides directes de la terminal durant les diferents execucions del codi per facilitar-ne la traçabilitat.

