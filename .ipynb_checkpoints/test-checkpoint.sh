curl -d '{"sepal_length":"5.1","sepal_width":"3.5","petal_length":"1.4","petal_width":"0.2"}' -X POST 127.0.0.1:5000/api/v1/predict -H "Content-Type: application/json"
echo ""
curl -d '{"sepal_length":"7.0","sepal_width":"3.2","petal_length":"4.7","petal_width":"1.4"}' -X POST 127.0.0.1:5000/api/v1/predict -H "Content-Type: application/json"
echo ""
curl -d '{"sepal_length":"6.3","sepal_width":"3.3","petal_length":"6.0","petal_width":"2.5"}' -X POST 127.0.0.1:5000/api/v1/predict -H "Content-Type: application/json"
echo ""
curl -d '{"sepal_length":"1.0","sepal_width":"3.5","petal_length":"1.4","petal_width":"0.2"}' -X POST 127.0.0.1:5000/api/v1/predict -H "Content-Type: application/json"
echo ""