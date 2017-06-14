#version 330
#ifdef GL_ES
    precision highp float;
#endif
in vec2 fragTexCoord;

uniform sampler2D tex0;
uniform int mipLevelIndex;

// Force location to 0 to ensure its the first output
layout (location = 0) out vec4 FragColor;

uniform sampler2D tex0Texture2DFramebufferTexture2D;

void main() {	
	
	/* 
	couple solution for mip mapping:
	* nearest: 1 and nearest
	* box filter 2x2 without linear interpolation: 4, nearest
	* box filter 4x4 with linear interpolation: 4, linear
	* quincunx: 5, nearest
	* Gaussian' 4x4 samples from box filter 4x4 moved to the center to have right weights: 4, linear
	* something with 5 samples and linear interpolation
	*/

	if (mipLevelIndex == 0) {
		FragColor = texture(tex0, fragTexCoord);
		return;
	}

	//FragColor = textureLod(tex0Texture2DFramebufferTexture2D, fragTexCoord, mipLevelIndex - 1); return;
	
	// probably the correct kernel
	vec2 coord = fragTexCoord;
	
	float x = 1.0 / textureSize(tex0Texture2DFramebufferTexture2D, mipLevelIndex - 1).x;
	float y = 1.0 / textureSize(tex0Texture2DFramebufferTexture2D, mipLevelIndex - 1).y;
	
	const int kernelSize = 2;
	float kernel[] = float [](1, 4, 6, 4, 1);
	//const int kernelSize = 3;
	//float kernel[] = float [](1, 6, 15, 20, 15, 6, 1);
	
	vec4 value = vec4(0);
	float weight = 0;
	for(int i = -kernelSize; i <= kernelSize; i++){
		for(int j = -kernelSize; j <= kernelSize; j++){
			float k = kernel[i + kernelSize] * kernel[j + kernelSize];
			value += textureLod(tex0Texture2DFramebufferTexture2D, coord + vec2(i * x, j * y), mipLevelIndex - 1) * k;
			weight += k;
		}
	}
	value = value / weight;
	FragColor = value;

	
	
    
}
