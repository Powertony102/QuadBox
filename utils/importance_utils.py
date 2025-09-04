import math
import torch    
import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy
from arguments import OptimizationParams, PipelineParams, ImportanceParams

class TileImportance():
    def __init__(self, viewpoint_stack, viewpoint_indices, importance_params=None):
        self.viewpoint_stack = viewpoint_stack
        self.viewpoint_indices = viewpoint_indices
        
        # 使用传入的参数或默认值
        if importance_params is not None:
            self.tile_size = importance_params.tile_size
            self.w_edge = importance_params.w_edge
            self.w_entropy = importance_params.w_entropy
            self.w_glcm = importance_params.w_glcm
        else:
            # 使用默认参数值（与ImportanceParams中的默认值保持一致）
            self.tile_size = 16
            self.w_edge = 0.4      # 边缘密度权重
            self.w_entropy = 0.3   # 香农熵权重
            self.w_glcm = 0.3      # GLCM对比度权重
        
    def compute_importance(self):
        """
        计算所有视图中所有瓦块的复杂度分数
        返回: dict[viewpoint_idx] = dict[tile_idx] = complexity_score
        """
        importance_scores = {}
        
        for i, viewpoint in enumerate(self.viewpoint_stack):
            # 获取原始图像
            image = viewpoint.original_image.cpu().numpy()
            if image.shape[0] == 3:  # RGB图像
                image = np.transpose(image, (1, 2, 0))
                # 转换为灰度图
                gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray_image = image.squeeze()
            
            # 将图像分割为瓦块
            tiles = self._split_into_tiles(gray_image)
            
            # 计算每个瓦块的复杂度分数
            tile_scores = {}
            for tile_idx, tile in enumerate(tiles):
                if tile is not None:
                    complexity_score = self._compute_tile_complexity(tile)
                    tile_scores[tile_idx] = complexity_score
            
            importance_scores[i] = tile_scores
            
        return importance_scores
    
    def _split_into_tiles(self, image):
        """
        将图像分割为瓦块
        """
        height, width = image.shape
        tiles = []
        
        for y in range(0, height, self.tile_size):
            for x in range(0, width, self.tile_size):
                # 确保瓦块不超出图像边界
                tile = image[y:y+self.tile_size, x:x+self.tile_size]
                if tile.shape == (self.tile_size, self.tile_size):
                    tiles.append(tile)
                else:
                    # 如果瓦块不完整，用零填充
                    padded_tile = np.zeros((self.tile_size, self.tile_size), dtype=image.dtype)
                    padded_tile[:tile.shape[0], :tile.shape[1]] = tile
                    tiles.append(padded_tile)
        
        return tiles
    
    def _compute_tile_complexity(self, tile):
        """
        计算单个瓦块的混合复杂度分数
        C(T_i) = w_edge * C_edge(T_i) + w_entropy * C_entropy(T_i) + w_glcm * C_glcm(T_i)
        """
        # 计算边缘密度
        edge_score = self._compute_edge_density(tile)
        
        # 计算香农熵
        entropy_score = self._compute_entropy(tile)
        
        # 计算GLCM对比度
        glcm_score = self._compute_glcm_contrast(tile)
        
        # 混合复杂度分数
        complexity_score = (self.w_edge * edge_score + 
                          self.w_entropy * entropy_score + 
                          self.w_glcm * glcm_score)
        
        return complexity_score
    
    def _compute_edge_density(self, tile):
        """
        计算边缘密度 - 使用Canny边缘检测
        """
        # 归一化到0-255范围
        tile_normalized = ((tile - tile.min()) / (tile.max() - tile.min() + 1e-8) * 255).astype(np.uint8)
        
        # 使用Canny边缘检测
        edges = cv2.Canny(tile_normalized, 50, 150)
        
        # 计算边缘像素密度
        edge_density = np.sum(edges > 0) / (self.tile_size * self.tile_size)
        
        return edge_density
    
    def _compute_entropy(self, tile):
        """
        计算香农熵
        """
        # 归一化到0-255范围
        tile_normalized = ((tile - tile.min()) / (tile.max() - tile.min() + 1e-8) * 255).astype(np.uint8)
        
        # 计算香农熵
        entropy = shannon_entropy(tile_normalized)
        
        # 归一化到[0,1]范围 (香农熵的最大值约为8)
        normalized_entropy = entropy / 8.0
        
        return min(normalized_entropy, 1.0)
    
    def _compute_glcm_contrast(self, tile):
        """
        计算GLCM对比度特征
        """
        # 归一化到0-255范围并量化为更少的灰度级
        tile_normalized = ((tile - tile.min()) / (tile.max() - tile.min() + 1e-8) * 255).astype(np.uint8)
        tile_quantized = (tile_normalized // 32) * 32  # 量化为8个灰度级
        
        try:
            # 计算GLCM
            glcm = graycomatrix(tile_quantized, distances=[1], angles=[0, 45, 90, 135], 
                              levels=8, symmetric=True, normed=True)
            
            # 计算对比度
            contrast = graycoprops(glcm, 'contrast').mean()
            
            # 归一化到[0,1]范围
            normalized_contrast = min(contrast / 100.0, 1.0)
            
            return normalized_contrast
        except:
            # 如果GLCM计算失败，返回0
            return 0.0


def precompute_importance(viewpoint_stack, viewpoint_indices, importance_params=None):
    """
    预计算所有视图中所有瓦块的重要性分数
    这个函数在训练开始前调用，用于一次性计算所有复杂度分数
    
    Args:
        viewpoint_stack: 视角列表
        viewpoint_indices: 视角索引列表
        importance_params: 重要性计算参数，可选
    """
    importance_calculator = TileImportance(viewpoint_stack, viewpoint_indices, importance_params)
    return importance_calculator.compute_importance()