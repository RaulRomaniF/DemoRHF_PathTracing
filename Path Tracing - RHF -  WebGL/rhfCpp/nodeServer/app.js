
// http://nodejs.org/api.html#_child_processes
// var sys  = require('sys')
// var exec = require('child_process').exec;
// var child;
// // executes `pwd`
// child = exec("pwd", function (error, stdout, stderr) {
//   sys.print('stdout: ' + stdout);
//   sys.print('stderr: ' + stderr);
//   if (error !== null) {
//     console.log('exec error: ' + error);
//   }
// });
// or more concisely
var sys  = require('sys')
var exec = require('child_process').exec;
function puts(error, stdout, stderr) { 
	sys.puts(stdout) 
	if (error !== null) {
	  console.log('exec error: ' + error);
	}
}
// exec("ls -la", puts);

exec("bash scripts.sh", puts);

