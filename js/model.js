let video;
let poseNet;
let pose;
let skeleton;

let brain;

let state = 'waiting';
let targeLabel;


function setup() {
  createCanvas(640, 480);
  
  video = createCapture(VIDEO);
  video.hide();
  poseNet = ml5.poseNet(video, modelLoaded);
  poseNet.on('pose', gotPoses);

  let options = {
    inputs: 34,
    outputs: 4,
    task: 'classification',
    debug: true
  }
  brain = ml5.neuralNetwork(options);
  const modelInfo = {
    model: 'model/model.json',
    metadata: 'model/model_meta.json',
    weights: 'model/model.weights.bin'
  }
  brain.load(modelInfo, brainLoaded);
  
  //train model
  // brain.loadData('model/bodybuilding-poses.json', dataReady);
}

function brainLoaded() {
  console.log('pose classification ready');
  classifyPose();
}

function classifyPose(){
  if(pose){
    let inputs = [];
    for (let i = 0; i < pose.keypoints.length; i++) {
      let x = pose.keypoints[i].position.x;
      let y = pose.keypoints[i].position.y;
      inputs.push(x);
      inputs.push(y);
    }
    brain.classify(inputs, gotResults)
  } else {
    //als er geen pose is gedetecteerd, probeer opniniuew na 100ms
    console.log("no pose")
    setTimeout(classifyPose, 100);
  }
}

function gotResults(error, results) {
  let poseName = document.getElementById('poseName')
  let confidence = document.getElementById('confidence')
  // console.log(results);
  if(results[0].label == 'a'){
    poseName.innerHTML = 'Relaxed Pose'
    confidence.innerHTML = Math.round(results[0].confidence * 100).toFixed(0) + '%'	
    console.log('relaxed pose');
  }
  if(results[0].label == 'b'){
    poseName.innerHTML = 'Front double biceps pose'
    confidence.innerHTML = Math.round(results[0].confidence * 100).toFixed(0) + '%'	
    console.log('front double biceps pose');
  }
  if(results[0].label == 'c'){
    poseName.innerHTML = 'Most muscular pose'
    confidence.innerHTML = Math.round(results[0].confidence * 100).toFixed(0) + '%'	
    console.log('most muscular pose');
  }
  if(results[0].label == 'd'){
    poseName.innerHTML = 'Front lat spread pose'
    confidence.innerHTML = Math.round(results[0].confidence * 100).toFixed(0) + '%'	
    console.log('front lat spread pose');
  }
  classifyPose(); 
}

function dataReady(){
  brain.normalizeData();
  brain.train({epochs:50}, finished);
}

function finished(){
  console.log('model trained');
  brain.save()
}


function gotPoses(poses) {
  // console.log(poses); 
  if (poses.length > 0) {
    pose = poses[0].pose;
    skeleton = poses[0].skeleton;
    if (state == 'collecting') {

      let inputs = [];
      for (let i = 0; i < pose.keypoints.length; i++) {
        let x = pose.keypoints[i].position.x;
        let y = pose.keypoints[i].position.y;
        inputs.push(x);
        inputs.push(y);
      }
      let target = [targetLabel];
      brain.addData(inputs, target);
    }
  }
}


function modelLoaded() {
  console.log('poseNet ready');
}

function draw() {
  translate(video.width, 0);
  scale(-1, 1);
  image(video, 0, 0, video.width, video.height);

  if (pose) {
    for (let i = 0; i < skeleton.length; i++) {
      let a = skeleton[i][0];
      let b = skeleton[i][1];
      strokeWeight(2);
      stroke(0);

      line(a.position.x, a.position.y, b.position.x, b.position.y);
    }
    for (let i = 0; i < pose.keypoints.length; i++) {
      let x = pose.keypoints[i].position.x;
      let y = pose.keypoints[i].position.y;
      fill(0);
      stroke(255);
      ellipse(x, y, 16, 16);
    }
  }
}