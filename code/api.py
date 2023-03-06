from flask import Flask
from flask_restful import Resource,Api, reqparse,abort
from experiment import *

app = Flask(__name__)
api = Api(app)


predict_parser = reqparse.RequestParser()
predict_parser.add_argument("name",type=str,required=True)
predict_parser.add_argument("url",type=str,required=True)
predict_parser.add_argument("model_number",type=int,required=True)


class HelloWorld(Resource):
    def get(self):
        return {'data':'hello world!'}

class Predict(Resource):
  def get(self):
     return {'data':'post request with name,url,model_number please'}
  def post(self):
    args = predict_parser.parse_args()
    if args['name'] == '' or args['url'] == '':  
      abort(400,"Name and url must be provided")
    else:
       result = predict(args['name'],args['url'],args['model_number'])
       return args



face_detection_parser = reqparse.RequestParser()
face_detection_parser.add_argument("url",type=str,required=True)
class FaceDetection(Resource):
   def post(self):
      args = face_detection_parser.parse_args()
      if args['url'] == '':
        abort(400,"url must be provided")
      else:
        result = detect_face(args['url'])
        if result == 1:
           return {'data':'true'}
        else:
           return {'data':'false'}
        



api.add_resource(HelloWorld, '/')
api.add_resource(Predict, '/predict')
api.add_resource(FaceDetection, '/face')



if __name__ == '__main__':
    app.run(debug=True)