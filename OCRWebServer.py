from flask import Flask, request, render_template, redirect, flash
from werkzeug.utils import secure_filename
import imutils
from imutils.object_detection import non_max_suppression
import numpy as np
import pytesseract
import os
import pprint
import cv2
import uuid
import json


app = Flask(__name__,static_url_path='/static')#to find images
app.secret_key = 'cooleschule'

EAST_TEXT_DETECTOR = 'east-text-detection/frozen_east_text_detection.pb'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
UPLOAD_FOLDER = os.path.basename('uploads')
IMAGE_FOLDER = 'static/images/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['IMAGE_FOLDER'] = IMAGE_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET','POST'])
def upload_file():

    if 'image' not in request.files:
            flash('No file part')
            return redirect('/')

    file = request.files['image']
    
    if file.filename == '':
            flash('No selected file')
            return redirect('/')
    if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return ocr_detection(os.path.join(app.config['UPLOAD_FOLDER'], filename),float(request.form.get("minConfidence", False)),int(request.form.get("width", False)),int(request.form.get("height", False)),float(request.form.get("padding", False)),request.form.get("language", False),int(request.form.get("oem", False)), int(request.form.get("psm", False)))

    return render_template('index.html')

@app.route('/ocrjson', methods=['GET','POST'])
def ocrjson():

    params = json.loads(request.form.get("json", False))
    if len(params) < 7:
        return 'not all parameters were specified'
    for key in params.keys():
        if params[key] == None:
            return 'at least one parameter is None'

    file = request.files['image']
    if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return ocr_detection(os.path.join(app.config['UPLOAD_FOLDER'], filename),float(params["minConfidence"]),int(params["width"]),int(params["height"]),float(params["padding"]),params["language"],int(params["oem"]), int(params["psm"]),isJson=True)

def allowed_file(filename):
    ''' Check if the file extension is allowed '''
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def decode_predictions(scores, geometry, minConfidence):
	# grab the number of rows and columns from the scores volume, then
	# initialize our set of bounding box rectangles and corresponding
	# confidence scores
	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []
 
	# loop over the number of rows
	for y in range(0, numRows):
		# extract the scores (probabilities), followed by the
		# geometrical data used to derive potential bounding box
		# coordinates that surround text
		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]
 
		# loop over the number of columns
		for x in range(0, numCols):
			# if our score does not have sufficient probability,
			# ignore it
			if scoresData[x] < minConfidence:
				continue
 
			# compute the offset factor as our resulting feature
			# maps will be 4x smaller than the input image
			(offsetX, offsetY) = (x * 4.0, y * 4.0)
 
			# extract the rotation angle for the prediction and
			# then compute the sin and cosine
			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)
 
			# use the geometry volume to derive the width and height
			# of the bounding box
			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]
 
			# compute both the starting and ending (x, y)-coordinates
			# for the text prediction bounding box
			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)
 
			# add the bounding box coordinates and probability score
			# to our respective lists
			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])
			#print('confidence: ' + str(scoresData[x]))
 
	# return a tuple of the bounding boxes and associated confidences
	return (rects, confidences)

def ocr_detection(filename, minConfidence, width, height, padding, language, oem, psm, isJson=False):
    # load the input image and grab the image dimensions
    image = cv2.imread(filename)
    orig = image.copy()
    (origH, origW) = image.shape[:2]
    
    # set the new width and height and then determine the ratio in change
    # for both the width and height
    (newW, newH) = (width, height)
    rW = origW / float(newW)
    rH = origH / float(newH)
    
    # resize the image and grab the new image dimensions
    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]
    
    # define the two output layer names for the EAST detector model that
    # we are interested in -- the first is the output probabilities and the
    # second can be used to derive the bounding box coordinates of text
    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]
    
    # load the pre-trained EAST text detector
    net = cv2.dnn.readNet(EAST_TEXT_DETECTOR)

    # construct a blob from the image and then perform a forward pass of
    # the model to obtain the two output layer sets
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
        (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)
    
    # decode the predictions, then  apply non-maxima suppression to
    # suppress weak, overlapping bounding boxes
    (rects, confidences) = decode_predictions(scores, geometry,minConfidence)
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    # initialize the list of results
    results = []
    
    # loop over the bounding boxes
    for (startX, startY, endX, endY) in boxes:
        # scale the bounding box coordinates based on the respective
        # ratios
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)
    
        # in order to obtain a better OCR of the text we can potentially
        # apply a bit of padding surrounding the bounding box -- here we
        # are computing the deltas in both the x and y directions
        dX = int((endX - startX) * padding)
        dY = int((endY - startY) * padding)
    
        # apply padding to each side of the bounding box, respectively
        startX = max(0, startX - dX)
        startY = max(0, startY - dY)
        endX = min(origW, endX + (dX * 2))
        endY = min(origH, endY + (dY * 2))
    
        # extract the actual padded ROI
        roi = orig[startY:endY, startX:endX]

        # in order to apply Tesseract v4 to OCR text we must supply
        # (1) a language, (2) an OEM flag of 4, indicating that the we
        # wish to use the LSTM neural net model for OCR, and finally
        # (3) an OEM value, in this case, 7 which implies that we are
        # treating the ROI as a single line of text
        config = ('-l ' + language + ' --oem ' + str(oem) + ' --psm ' + str(psm)) 
        text = pytesseract.image_to_string(roi, config=config)
    
        # add the bounding box coordinates and OCR'd text to the list
        # of results
        results.append(((startX, startY, endX, endY), text))

        # sort the results bounding box coordinates from top to bottom
    results = sorted(results, key=lambda r:r[0][1])

    #return text
    remove_contents(app.config['IMAGE_FOLDER'])

    images = []
    textToShow = ''
    lastsettings = {'minConfidence':minConfidence, 'width':width, 'height':height, 'padding':padding, 'language':language, 'oem' : oem, 'psm':psm}
    # loop over the results
    for ((startX, startY, endX, endY), text) in results:
    
        # strip out non-ASCII text so we can draw the text on the image
        # using OpenCV, then draw the text and a bounding box surrounding
        # the text region of the input image
        text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
        textToShow += ' ' + text
        output = orig.copy()
        #print(text)
        #print(output)
        cv2.rectangle(output, (startX, startY), (endX, endY),(0, 0, 255), 2)
        cv2.putText(output, text, (startX, startY - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        
        filepath = app.config['IMAGE_FOLDER'] + str(uuid.uuid4().hex) + '.png'
        cv2.imwrite(filepath, output)
        images.append(filepath)

    if isJson:
        return textToShow
    else:
        return render_template('index.html', text=textToShow, images=images, lastsettings=lastsettings)

def remove_contents(path):
    for c in os.listdir(path):
        full_path = os.path.join(path, c)
        if os.path.isfile(full_path):
            os.remove(full_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000,threaded=True, debug=False) #to make it visible 10.100.x.x only for 10.100.0.0 /16 
