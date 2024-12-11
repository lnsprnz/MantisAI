Bei meinem Ansatz müssten wir zum Training Bilder + ein COCO Annotation File verwenden. In dem File werden die Keypoints definiert (z.B. die welche Fabian schon einmal gesetzt hat oder über ein spezifisches Annotation Tool) 
Der beste Weg ist, dass wir Fabian Keypoint verwenden und mit einem kleinen Skript in das entsprechende Format pressen. 

Falls wir die Keypoints fürs Training nochmal "perfekt" setzen wollen können wir Annotation Tools verwenden ich habe mir bisher  zwei Keypoint Annotation Tool angeschaut:
- Roboflow (https://app.roboflow.com): Ein ganz schickes Tool/bzw. Plattform für kollaboratives annotieren von Daten und viele weitere Sachen, teilweise kostenpflichtig.
- COCO Annotator (https://github.com/jsbroks/coco-annotator/wiki): Das Ding würde ich gerne nutzen, ich bekomme das aber noch nicht ganz ans Laufen wegen ein paar Docker Struggel bei Windows. Pros: Open Source, kollaboratives Arbeiten auch möglich. 

Mit Roboflow habe ich einfach mal ein Bild annotiert. Das entsprechende Bild + File findet ihr in meinem Arbeitsordner
