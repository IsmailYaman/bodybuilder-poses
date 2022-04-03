let brain;

function setup() {
  createCanvas(640, 480);
  let options = {
    inputs: 34,
    outputs: 4,
    task: 'classification',
    debug: true
  }
  brain = ml5.neuralNetwork(options);
  const modelInfo = {
    model: 'model/model.json',
    metadata: 'model/model-meta.json',
    weights: 'model/model.weights.bin'
  }
  brain.load(modelInfo, modelReady);
  brain.loadData('./bodybuilder-poses.json', dataReady);
}

function modelReady(){
  console.log('model ready');
}
function dataReady() {
  brain.normalizeData();
  brain.train({epochs: 50}, finished); 
}

function finished() {
  console.log('model trained');
  brain.save();
}