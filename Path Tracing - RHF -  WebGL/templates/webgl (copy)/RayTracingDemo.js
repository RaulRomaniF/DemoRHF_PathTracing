'use strict';

var RayTracingDemo = function(gl) {
	this.gl = gl;
};


RayTracingDemo.prototype.Load = function (cb) {
	console.log('Loading demo scene');

	var me = this;


	async.parallel({
		Models: function (callback) {
			async.map({
				CubeModelText: 'models/cube.obj'
			}, LoadTextResource, callback);
		},
		ShaderCode: function (callback) {
			async.map({
				'PathTracer_VSText': 'shaders/pathTracer.vs.glsl',
				'PathTracer_FSText': 'shaders/pathTracer.fs.glsl',
			}, LoadTextResource, callback);
		}
	}, function (loadErrors, loadResults) {
		if (loadErrors) {
			cb(loadErrors);
			return;
		}


		//
		// Create Shaders
		//
		me.PathTracerProgram = CreateShaderProgram(
			me.gl, 
			loadResults.ShaderCode.PathTracer_VSText,
			loadResults.ShaderCode.PathTracer_FSText
		);

		if (me.PathTracerProgram.error) {
			cb('PathTracerProgram ' + me.PathTracerProgram.error); return;
		}


		//
	    // Create buffer for quads
	    //
		var quadVerts = 
		[ 
			-1.0, -1.0,
			 1.0, -1.0,
			 1.0,  1.0,
			-1.0,  1.0
		];
		
		//setup quad indices
		var quadIndices =
		[
			0, 1, 2,
			0, 2, 3,
		];

		//setup quad vertex array and vertex buffer objects
		me.quadVAOID     = me.gl.createVertexArray();
		var quadVBOID     = me.gl.createBuffer();
		var quadIndicesID = me.gl.createBuffer();


		me.gl.bindVertexArray(me.quadVAOID);
			me.gl.bindBuffer (me.gl.ARRAY_BUFFER, quadVBOID);
				//pass quad vertices to vertex buffer object
				me.gl.bufferData (me.gl.ARRAY_BUFFER, new Float32Array(quadVerts), me.gl.STATIC_DRAW);

				//enable vertex attribute array for vertex position
				var vertexAttribLocation = me.gl.getAttribLocation(me.PathTracerProgram, 'vPos');
				me.gl.enableVertexAttribArray(vertexAttribLocation);		
				me.gl.vertexAttribPointer(
					vertexAttribLocation, // Attribute location
					2, // Number of elements per attribute
					me.gl.FLOAT, // Type of elements
					me.gl.FALSE,
					2 * Float32Array.BYTES_PER_ELEMENT, // Size of an individual vertex
					0  // Offset from the beginning of a single vertex to this attribute
				);

			//pass quad indices to element array buffer
			me.gl.bindBuffer(me.gl.ELEMENT_ARRAY_BUFFER, quadIndicesID);
				me.gl.bufferData(me.gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(quadIndices), me.gl.STATIC_DRAW);
		me.gl.bindVertexArray(null);
		

		//
		// Create Model objects
		//

		console.log("model lenght"+ loadResults.Models.CubeModelText.length);
		var bloquesModel = ObjLoader.parseObjText( loadResults.Models.CubeModelText , false);

	    var vertices = bloquesModel[0];
		var indices  = bloquesModel[1];
	    var aabbMin  = bloquesModel[2];
	    var aabbMax  = bloquesModel[3];

	    console.log("v" +parseFloat(vertices.length/3));
		console.log("i"+ parseFloat(indices.length/4));

	    //
		// SET UNIFORMS
		//
		me.gl.useProgram(me.PathTracerProgram);
       	me.gl.uniform1f( me.gl.getUniformLocation(me.PathTracerProgram, "VERTEX_TEXTURE_SIZE"), parseFloat(vertices.length/3));
		me.gl.uniform1f( me.gl.getUniformLocation(me.PathTracerProgram, "TRIANGLE_TEXTURE_SIZE"), parseFloat(indices.length)/4);

		me.gl.uniform3fv(me.gl.getUniformLocation(me.PathTracerProgram, "aabb.min"), [aabbMin[0], aabbMin[1], aabbMin[2] ]);
		me.gl.uniform3fv(me.gl.getUniformLocation(me.PathTracerProgram, "aabb.max"), [aabbMax[0], aabbMax[1], aabbMax[2] ]);

		me.gl.uniform4fv(me.gl.getUniformLocation(me.PathTracerProgram, "backgroundColor"), [0.5, 0.5, 0.5, 1]);
		me.gl.uniform1i( me.gl.getUniformLocation(me.PathTracerProgram, "vertex_positions"), 1);
		me.gl.uniform1i( me.gl.getUniformLocation(me.PathTracerProgram, "triangles_list")  , 2);


		//
		// LOAD TEXTURES - pass position to 1D texture bound to texture unit 1
		//
		me.texVerticesID = me.gl.createTexture();
		me.gl.activeTexture(me.gl.TEXTURE1);
		me.gl.bindTexture(me.gl.TEXTURE_2D, me.texVerticesID);
		me.gl.texParameteri(me.gl.TEXTURE_2D, me.gl.TEXTURE_WRAP_S, me.gl.CLAMP_TO_EDGE);
		me.gl.texParameteri(me.gl.TEXTURE_2D, me.gl.TEXTURE_WRAP_T, me.gl.CLAMP_TO_EDGE);
		me.gl.texParameteri(me.gl.TEXTURE_2D, me.gl.TEXTURE_MIN_FILTER, me.gl.NEAREST);
		me.gl.texParameteri(me.gl.TEXTURE_2D, me.gl.TEXTURE_MAG_FILTER, me.gl.NEAREST);

		console.log("vertices:" + vertices.length)
		var pData = new Float32Array(12*4);
		var count = 0;
		for(var i=0; i<vertices.length; i++) {
			pData[count++] = vertices[i*3 +0];
			pData[count++] = vertices[i*3 +1];
			pData[count++] = vertices[i*3 +2];
			pData[count++] = 0;
		}
		
		me.gl.texImage2D(me.gl.TEXTURE_2D, 0, me.gl.RGBA32F,
	                12, 1, 0,
	                me.gl.RGBA, me.gl.FLOAT,  pData);

		pData = null;
		me.gl.bindTexture(me.gl.TEXTURE_2D, null);

		//
		// store the mesh topology in another texture bound to texture unit 2
		//
		me.texTrianglesID = me.gl.createTexture();
		me.gl.activeTexture(me.gl.TEXTURE2);
		me.gl.bindTexture(me.gl.TEXTURE_2D, me.texTrianglesID);
		me.gl.texParameteri(me.gl.TEXTURE_2D, me.gl.TEXTURE_WRAP_S, me.gl.CLAMP_TO_EDGE);
		me.gl.texParameteri(me.gl.TEXTURE_2D, me.gl.TEXTURE_WRAP_T, me.gl.CLAMP_TO_EDGE);
		me.gl.texParameteri(me.gl.TEXTURE_2D, me.gl.TEXTURE_MIN_FILTER, me.gl.NEAREST);
		me.gl.texParameteri(me.gl.TEXTURE_2D, me.gl.TEXTURE_MAG_FILTER, me.gl.NEAREST);


		console.log("TRIANGLES:" + indices.length/4)
		var pData = new Int32Array(indices.length);
		count = 0;
		for(var i=0; i<indices.length; i+=4) {
			pData[count++] = indices[i+0];
			pData[count++] = indices[i+1];
			pData[count++] = indices[i+2];
			pData[count++] = indices[i+3];
		}

		me.gl.texImage2D(me.gl.TEXTURE_2D, 0, me.gl.RGBA32I,
	                indices.length/4, 1, 0,
	                me.gl.RGBA_INTEGER, me.gl.INT,  pData);

		pData = null;
		me.gl.bindTexture(me.gl.TEXTURE_2D, null);

		//set texture unit 0 as active texture unit
		me.gl.activeTexture(me.gl.TEXTURE0);

		//
		// Logical Values
		//
		me.width = 800;
		me.height = 600;

		//set the size of the canvas, on chrome we need to set it 3 ways to make it work perfectly.
		me.gl.canvas.style.width = me.width + "px";
		me.gl.canvas.style.height = me.height + "px";
		me.gl.canvas.width = me.width;
		me.gl.canvas.height = me.height;

		//when updating the canvas size, must reset the viewport of the canvas 
		//else the resolution webgl renders at will not change
		me.gl.viewport(0,0,me.width, me.height);

		me.camera = new Camera(
			// vec3.fromValues(15.0,  5.0, -10.0),  //eyepos
			vec3.fromValues(40.936004638671875, 14.22861099243164, -31.0819149017334),  //eyepos
			
			vec3.fromValues(0, 0, 0),
			vec3.fromValues(0, 1, 0)
		);

		me.projMatrix = mat4.create();
		me.viewMatrix = mat4.create();

		mat4.perspective(
			me.projMatrix,
			glMatrix.toRadian(60),
			me.width / me.height,
			0.35,
			800.0
		);

		me.lightPosition = vec3.fromValues(55.0, 35.0, -10.0);


		cb();
	});

	me.PressedKeys = {
		Up: false,
		Right: false,
		Down: false,
		Left: false,
		Forward: false,
		Back: false,

		RotLeft: false,
		RotRight: false,

		Pause: false,

	};

	me.MoveForwardSpeed = 14.5;
	me.RotateSpeed = 1.5;
	// me.textureSize = getParameterByName('texSize') || 512;
	me.lightDisplacementInputAngle = 0.0;
};

RayTracingDemo.prototype.Unload = function () {
	// this.LightMesh = null;
	// this.MonkeyMesh = null;
	// this.TableMesh = null;
	// this.SofaMesh = null;
	// this.WallsMesh = null;

	// this.PathTracerProgram = null;
	// this.ShadowProgram = null;
	// this.ShadowMapGenProgram = null;

	// this.camera = null;
	// this.lightPosition = null;

	// this.Meshes = null;

	// this.PressedKeys = null;

	// this.MoveForwardSpeed = null;
	// this.RotateSpeed = null;

	// this.shadowMapCube = null;
	// this.textureSize = null;

	// this.shadowMapCameras = null;
	// this.shadowMapViewMatrices = null;
};

RayTracingDemo.prototype.Begin = function () {
	console.log('Beginning demo scene');

	var me = this;

	// Attach event listeners
	this.__KeyDownWindowListener = this._OnKeyDown.bind(this);
	this.__KeyUpWindowListener = this._OnKeyUp.bind(this);

	AddEvent(window, 'keydown', this.__KeyDownWindowListener);
	AddEvent(window, 'keyup', this.__KeyUpWindowListener);
	
	// Render Loop
	var previousFrame = performance.now();
	var dt = 0;
	var loop = function (currentFrameTime) {
		dt = currentFrameTime - previousFrame;
		me._Update(dt);
		previousFrame = currentFrameTime;

		me._Render();
		me.nextFrameHandle = requestAnimationFrame(loop);

		if (me.PressedKeys.Pause ) {
			// window.cancelAnimationFrame(this.nextFrameHandle);
			me.DrawPixelsBuffer();


			cancelAnimationFrame(me.nextFrameHandle);
			console.log("pause pressed");


		}
	};
	me.nextFrameHandle = requestAnimationFrame(loop);

	

	


	// me._OnResizeWindow();
};

RayTracingDemo.prototype.DrawPixelsBuffer = function () {
	console.log('Beginning DrawPixelsBuffer');

	var gl = this.gl;

	var pixels = new Uint8Array(gl.drawingBufferWidth * gl.drawingBufferHeight * 4);
	gl.readPixels(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight, gl.RGBA, gl.UNSIGNED_BYTE, pixels);

	var canvas = document.createElement('canvas');
    var canvasContext=canvas.getContext("2d");


	canvas.width = 800;
	canvas.height = 600;


	var imageObj = new Image();
    canvasContext.drawImage(imageObj, 0, 0);

    var imageData = canvasContext.getImageData(0, 0, this.width, this.height);

   // This loop gets every pixels on the image and
    for (var j=0; j<imageData.height; j++)
    {
      for (var i=0; i<imageData.width; i++)
      {
         var offset=(j*4)*imageData.width + (i*4);

         var offset2= (imageData.height -1 - j)*imageData.width*4 + (i*4);
         
   	     imageData.data[offset]   = pixels[offset2];
         imageData.data[offset+1] = pixels[offset2+1];
         imageData.data[offset+2] = pixels[offset2+2];
         imageData.data[offset+3] = pixels[offset2+3];
       }
     }

    canvasContext.putImageData(imageData, 0, 0);
	document.getElementById("result").appendChild(canvas);
}


RayTracingDemo.prototype.End = function () {
	if (this.__ResizeWindowListener) {
		RemoveEvent(window, 'resize', this.__ResizeWindowListener);
	}
	if (this.__KeyUpWindowListener) {
		RemoveEvent(window, 'keyup', this.__KeyUpWindowListener);
	}
	if (this.__KeyDownWindowListener) {
		RemoveEvent(window, 'keydown', this.__KeyDownWindowListener);
	}

	if (this.nextFrameHandle) {
		cancelAnimationFrame(this.nextFrameHandle);
	}
};



//
// Private Methods
//
RayTracingDemo.prototype._Update = function (dt) {

	if (this.PressedKeys.Forward && !this.PressedKeys.Back) {
		this.camera.moveForward(dt / 1000 * this.MoveForwardSpeed);
	}

	if (this.PressedKeys.Back && !this.PressedKeys.Forward) {
		this.camera.moveForward(-dt / 1000 * this.MoveForwardSpeed);
	}

	if (this.PressedKeys.Right && !this.PressedKeys.Left) {
		this.camera.moveRight(dt / 1000 * this.MoveForwardSpeed);
	}

	if (this.PressedKeys.Left && !this.PressedKeys.Right) {
		this.camera.moveRight(-dt / 1000 * this.MoveForwardSpeed);
	}

	

	// if (this.PressedKeys.Up && !this.PressedKeys.Down) {
	// 	this.camera.moveUp(dt / 1000 * this.MoveForwardSpeed);
	// }

	// if (this.PressedKeys.Down && !this.PressedKeys.Up) {
	// 	this.camera.moveUp(-dt / 1000 * this.MoveForwardSpeed);
	// }

	// if (this.PressedKeys.RotRight && !this.PressedKeys.RotLeft) {
	// 	this.camera.rotateRight(-dt / 1000 * this.RotateSpeed);
	// }

	// if (this.PressedKeys.RotLeft && !this.PressedKeys.RotRight) {
	// 	this.camera.rotateRight(dt / 1000 * this.RotateSpeed);
	// }

	this.camera.GetViewMatrix(this.viewMatrix);
};


RayTracingDemo.prototype._Render = function () {
	var gl = this.gl;

	// Clear back buffer, set per-frame uniforms
	gl.enable(gl.DEPTH_TEST);
	gl.enable(gl.CULL_FACE);

	// gl.viewport(0, 0, gl.canvas.clientWidth, gl.canvas.clientHeight);

	    var MVP_INV = new Float32Array(16);
		var MVP = new Float32Array(16);
		mat4.multiply(MVP, this.projMatrix, this.viewMatrix);
		mat4.invert(MVP_INV, MVP);

		gl.clearColor(0.5, 0.5, 0.5, 1.0);
		gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

		var current = Math.random();
     	gl.useProgram(this.PathTracerProgram);

	    gl.uniform1i(gl.getUniformLocation(this.PathTracerProgram, "vertex_positions"), 1);
	    gl.uniform1i(gl.getUniformLocation(this.PathTracerProgram, "triangles_list"), 2); // set it manually

     	gl.activeTexture(gl.TEXTURE1);
     	gl.bindTexture(gl.TEXTURE_2D, this.texVerticesID);
		
		gl.activeTexture(gl.TEXTURE2);
		gl.bindTexture(gl.TEXTURE_2D, this.texTrianglesID);

     	gl.uniform3fv(gl.getUniformLocation(this.PathTracerProgram, "eyePos"), this.camera.position); //*************

		gl.uniform1f (gl.getUniformLocation(this.PathTracerProgram, "time"), current);
		gl.uniform3fv(gl.getUniformLocation(this.PathTracerProgram, "light_position"), this.lightPosition);
		gl.uniformMatrix4fv(gl.getUniformLocation(this.PathTracerProgram, "invMVP"), gl.FALSE, MVP_INV);

	    gl.bindVertexArray(this.quadVAOID);
		gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);



	// Draw meshes
};


RayTracingDemo.prototype._OnKeyDown = function (e) {
	switch(e.code) {
		case 'KeyW':
			this.PressedKeys.Forward = true;
			break;
		case 'KeyA':
			this.PressedKeys.Left = true;
			break;
		case 'KeyD':
			this.PressedKeys.Right = true;
			break;
		case 'KeyS':
			this.PressedKeys.Back = true;
			break;
		case 'KeyP':
			this.PressedKeys.Pause = true;
			break;
		// case 'ArrowUp':
		// 	this.PressedKeys.Up = true;
		// 	break;
		// case 'ArrowDown':
		// 	this.PressedKeys.Down = true;
		// 	break;
		// case 'ArrowRight':
		// 	this.PressedKeys.RotRight = true;
		// 	break;
		// case 'ArrowLeft':
		// 	this.PressedKeys.RotLeft = true;
		// 	break;
	}
};

RayTracingDemo.prototype._OnKeyUp = function (e) {
	switch(e.code) {
		case 'KeyW':
			this.PressedKeys.Forward = false;
			break;
		case 'KeyA':
			this.PressedKeys.Left = false;
			break;
		case 'KeyD':
			this.PressedKeys.Right = false;
			break;
		case 'KeyS':
			this.PressedKeys.Back = false;
			break;
		case 'KeyP':
			this.PressedKeys.Pause = false;
			break;

			
		// case 'ArrowUp':
		// 	this.PressedKeys.Up = false;
		// 	break;
		// case 'ArrowDown':
		// 	this.PressedKeys.Down = false;
		// 	break;
		// case 'ArrowRight':
		// 	this.PressedKeys.RotRight = false;
		// 	break;
		// case 'ArrowLeft':
		// 	this.PressedKeys.RotLeft = false;
		// 	break;
	}
};