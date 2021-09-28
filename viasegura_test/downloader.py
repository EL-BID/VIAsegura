from pathlib import Path
import boto3
import os
import sys

python_path = os.path.dirname(sys.executable)
library_path = '\\Lib\\site-packages\\viasegura_test\\'
viasegura_path = python_path+library_path
	

class Downloader:
	def __init__(self, models_path = viasegura_path+'models/'):
		self.models_path = models_path
	
	def check_artifacts(self):
		if not Path(self.models_path).is_dir():
			raise ImportError('The route for the models is not present, it means that the models are not downloaded on this environment, use viasegura.download_models function to download them propertly')
	
	def check_files(self, path):
		if Path(path).is_file():
			return True
		else:
			return False
	
	def download(self, aws_access_key=None, aws_secret_key=None):
		if not aws_access_key or not aws_secret_key:
			raise NameError('Must provide valid aws_access_key and aws_secret_key values')
		if not Path(self.models_path).is_dir():
			Path(self.models_path).mkdir(parents=True, exist_ok=True)
		sess = boto3.Session(aws_access_key_id = aws_access_key,
							aws_secret_access_key = aws_secret_key,
							region_name = 'us-east-1')
		s3 = sess.resource('s3')
		s3_client = sess.client('s3')
		my_bucket = s3.Bucket('via-segura-artifacts')
		for my_bucket_object in my_bucket.objects.all():
			elements = my_bucket_object.key.split('/')
			if elements[-1]=="":
				Path(self.models_path+my_bucket_object.key).mkdir(parents=True, exist_ok=True)
			else:
				my_file = Path(self.models_path+my_bucket_object.key)
				if not self.check_files(self.models_path+my_bucket_object.key):
					with open(self.models_path+my_bucket_object.key, 'wb') as f:
						s3_client.download_fileobj('via-segura-artifacts', my_bucket_object.key, f)
					
		print('Elements Downloaded')