'use strict';

var Demo;

function Init() {
	var canvas = document.getElementById('gl-surface');
	var gl = canvas.getContext('webgl2');
	if (!gl) {
		console.log('Failed to get WebGL context');
		gl = canvas.getContext('experimental-webgl');
	}
	if (!gl) {
		alert('Your browser does not support WebGL, use a different browser like Google Chrome or firefox');
		return;
	}

	Demo = new RayTracingDemo(gl);
	Demo.Load(function (demoLoadError) {
		if (demoLoadError) {
			alert('Could not load the demo, see console for more details');
			console.error(demoLoadError);
		} else {
			Demo.Begin();
		}
	});
}