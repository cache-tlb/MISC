const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const container = document.getElementById("container");

// states
var mouse_state = 0;    // 0: not pressed, 1: pressed
var mouse_x = 0;
var mouse_y = 0;



function init() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
}

function IsMousePressed() {
    return mouse_state == 1;
}


init();

function Vector2(x, y) {
    this.x = (x === undefined) ? 0 : x;
    this.y = (y === undefined) ? 0 : y;
}

Vector2.prototype = {
    add: function(vector) {
        return new Vector2(this.x + vector.x, this.y + vector.y);
    },

    mult: function(scalar) {
        return new Vector2(this.x * scalar, this.y * scalar);
    }
};

var index = 0;
var last = 0;

function line(ax, ay, bx, by) {
    ctx.beginPath();
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    ctx.lineWidth = 2;
    ctx.moveTo(ax*8, ay*8);
    ctx.lineTo(bx*8, by*8);
    ctx.stroke();
}
function v(x,y) {
    return new Vector2(x,y);
}

function h(i) {
    var a = v((i&3)>>1,[0,1,1][i&3]);
    for(var j = 1; j < 8;){
        var i=Math.floor(i/4);
        var k=i&3;
        var l=2**j;
        j++;
        !k?a=v(a.y,a.x):k<2?a.y+=l:k==2?a=a.add(v(l,l)):a=v(l-1-a.y+l,l-1-a.x)   
    }
    return a;
}

/*
i=n=a=0;draw=_=>{i||createCanvas(q=512,q);colorMode(HSB,1);v=createVector;h(i++);m=a.mult(4).add(2);stroke(i/4**7,1,1);line(m.x,m.y,n.x,n.y);n=m};h=i=>{a=v((i&3)>>1,[0,1,1][i&3]);for(j=1;j<8;){i/=4;k=i&3;l=2**j++;!k?a=v(a.y,a.x):k<2?a.y+=l:k==2?a.add(l,l):a=v(l-1-a.y+l,l-1-a.x)}}
*/
function draw() {
    var cur = h(index++);
    line(cur.x + 1,cur.y + 1, last.x + 1, last.y + 1);
    last = cur;
}

function render() {
    draw();
    window.requestAnimationFrame(render);
}
window.requestAnimationFrame(render);

canvas.addEventListener('mousedown', function (event) {
    mouse_state = 1;
    mouse_x = event.clientX;
    mouse_y = event.clientY;
}, false);
canvas.addEventListener('mousemove', function (event) {
    mouse_x = event.clientX;
    mouse_y = event.clientY;
}, false);
canvas.addEventListener('mouseup', function (event) {
    mouse_state = 0;
    mouse_x = event.clientX;
    mouse_y = event.clientY;
}, false);
function getTouchPos(canvasDom, touchEvent) {
  var rect = canvasDom.getBoundingClientRect();
  return {
    x: touchEvent.touches[0].clientX - rect.left,
    y: touchEvent.touches[0].clientY - rect.top
  };
}
canvas.addEventListener("touchstart", function (e) {
    mousePos = getTouchPos(canvas, e);
    var touch = e.touches[0];
    var mouseEvent = new MouseEvent("mousedown", {
        clientX: touch.clientX,
        clientY: touch.clientY
    });
  canvas.dispatchEvent(mouseEvent);
  event.preventDefault();
}, false);
canvas.addEventListener("touchend", function (e) {
    var mouseEvent = new MouseEvent("mouseup", {});
    canvas.dispatchEvent(mouseEvent);
    event.preventDefault();
}, false);
canvas.addEventListener("touchmove", function (e) {
    var touch = e.touches[0];
    var mouseEvent = new MouseEvent("mousemove", {
        clientX: touch.clientX,
        clientY: touch.clientY
    });
    canvas.dispatchEvent(mouseEvent);
    event.preventDefault();
}, false);


///
import GUI from './lil-gui.js';

const gui = new GUI( {container: document.getElementById( 'container' )});
gui.title('hahaha')

const folder = gui.addFolder( 'Folder' );

const folderParams = {
    number: 0.5,
    boolean: false,
    color: '#0cf',
    function() { console.log( 'hi' ) }
};

folder.add( folderParams, 'number', 0, 1 );
folder.add( folderParams, 'boolean' );
folder.addColor( folderParams, 'color' );
folder.add( folderParams, 'function' );

const params = {
    options: 10,
    boolean: true,
    string: 'lil-gui',
    number: 0,
    color: '#aa00ff',
    function() { console.log( 'hi' ) }
};

gui.add( params, 'options', { Small: 1, Medium: 10, Large: 100 } );
gui.add( params, 'boolean' );
gui.add( params, 'string' );
gui.add( params, 'number' );
gui.addColor( params, 'color' );
gui.add( params, 'function' ).name( 'Custom Name' );
