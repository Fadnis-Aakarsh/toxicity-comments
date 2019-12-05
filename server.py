from flask import Flask, request, Response
import json
import bert_toxicity

# Initialize the Flask application
app = Flask(__name__)

@app.route('/toxicity', methods=['GET'])
def getToxicity():
    text = request.args.get('text')
    input_format = [{
        'text': text,
        'label': '0'
    }]
    result = bert_toxicity.evaluate_exec(input_format, is_label_present=False)
    toxicity = 'N/A'
    if result and result[0] == 1:
        toxicity = 'toxic'
    else:
        toxicity = 'non_toxic'
    data = {
        'toxicity': toxicity
    }
    json_response = json.dumps(data)
    response = Response(json_response, status=200, mimetype='application/json')
    return response

# start flask app
app.run(host='0.0.0.0', port=5000)

