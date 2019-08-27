!#/bin/bash

wget https://dl.google.com/cloudsql/cloud_sql_proxy.linux.amd64 -O cloud_sql_proxy
chmod +x cloud_sql_proxy
sudo mkdir /cloudsql; sudo chmod 777 /cloudsql

./cloud_sql_proxy -instances=helpful-quanta-248212:us-central1:model-monitoring-1 -dir=/cloudsql
