from rest_framework import serializers

class SnakeRequestSerializer(serializers.Serializer):
    image = serializers.ImageField()
    contour_type = serializers.ChoiceField(choices=['manual'])
    manual_points = serializers.CharField(required=True)
    num_points = serializers.IntegerField(default=40, min_value=10, max_value=100)
    alpha = serializers.FloatField(default=0.3, min_value=0.0, max_value=1.0)
    beta = serializers.FloatField(default=0.5, min_value=0.0, max_value=1.0)
    gamma = serializers.FloatField(default=1.5, min_value=0.0, max_value=1.0)  
    max_iterations = serializers.IntegerField(default=200, min_value=10, max_value=500)

class SnakeResponseSerializer(serializers.Serializer):
    chain_code = serializers.ListField(child=serializers.IntegerField())
    perimeter = serializers.FloatField()
    area = serializers.FloatField()
    visualization = serializers.CharField()
    contour_points = serializers.ListField(child=serializers.ListField(child=serializers.FloatField()))
    convergence_history = serializers.ListField(child=serializers.FloatField())