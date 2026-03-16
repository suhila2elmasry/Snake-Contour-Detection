from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import cv2
import numpy as np
import os
import json
import traceback
from .serializers import SnakeRequestSerializer, SnakeResponseSerializer
from .snake_algorithm import GreedySnake

class SnakeProcessView(APIView):
    def post(self, request):
        try:
            # Validate request
            serializer = SnakeRequestSerializer(data=request.data)
            if not serializer.is_valid():
                print("Serializer errors:", serializer.errors)
                return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
            
            data = serializer.validated_data
            print("=" * 50)
            print("Processing snake request...")
            
            # Load image
            image_file = data['image']
            file_path = default_storage.save(f'uploads/{image_file.name}', ContentFile(image_file.read()))
            full_path = default_storage.path(file_path)
            print(f"Image saved to: {full_path}")
            
            # Read image with OpenCV
            image = cv2.imread(full_path)
            if image is None:
                return Response({'error': 'Could not read image'}, status=status.HTTP_400_BAD_REQUEST)
            
            # Get image dimensions
            height, width = image.shape[:2]
            print(f"Image dimensions: {width}x{height}")
            
            # Create initial contour from manual points
            contour_type = data.get('contour_type', 'manual')
            
            if contour_type == 'manual':
                # Parse manual points from JSON
                manual_points_json = data.get('manual_points')
                
                if not manual_points_json:
                    return Response({'error': 'No manual points provided'}, status=status.HTTP_400_BAD_REQUEST)
                
                try:
                    points_list = json.loads(manual_points_json)
                except json.JSONDecodeError as e:
                    return Response({'error': f'Invalid JSON format: {str(e)}'}, status=status.HTTP_400_BAD_REQUEST)
                
                print(f"Received {len(points_list)} manual points")
                
                if len(points_list) < 3:
                    return Response({'error': 'At least 3 points are required'}, status=status.HTTP_400_BAD_REQUEST)
                
                # Convert to numpy array
                initial_contour = np.array(points_list, dtype=np.float32)
                
                # Use points as is (no interpolation to avoid issues)
                print(f"Using manual contour with {len(initial_contour)} points")
            else:
                return Response({'error': 'Only manual contour type is supported'}, status=status.HTTP_400_BAD_REQUEST)
            
            # Initialize and evolve snake
            snake = GreedySnake(
                image=image,
                initial_contour=initial_contour,
                alpha=float(data.get('alpha', 0.3)),
                beta=float(data.get('beta', 0.5)),
                gamma=float(data.get('gamma', 0.5)),
                max_iterations=int(data.get('max_iterations', 200))
            )
            
            # Evolve snake
            print("Starting snake evolution...")
            final_contour = snake.evolve()
            
            # Get results
            chain_code = snake.get_chain_code()
            perimeter = snake.compute_perimeter()
            area = snake.compute_area()
            visualization = snake.get_visualization()
            
            # Clean up uploaded file
            if os.path.exists(full_path):
                os.remove(full_path)
                print(f"Cleaned up: {full_path}")
            
            # Prepare response
            response_data = {
                'chain_code': chain_code,
                'perimeter': float(perimeter),
                'area': float(area),
                'visualization': visualization,
                'contour_points': final_contour.tolist(),
                'convergence_history': snake.convergence_history
            }
            
            print(f"Response prepared: chain_code length={len(chain_code)}, perimeter={perimeter:.2f}, area={area:.2f}")
            print("=" * 50)
            
            return Response(response_data, status=status.HTTP_200_OK)
                
        except Exception as e:
            print("!" * 50)
            print("ERROR in snake processing:")
            traceback.print_exc()
            print("!" * 50)
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class HealthCheckView(APIView):
    def get(self, request):
        return Response({'status': 'healthy', 'message': 'Snake API is running'})