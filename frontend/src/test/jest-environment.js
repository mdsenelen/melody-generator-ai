// jsdom, plus Node's own fetch/Request/Response/FormData/Blob/File.
//
// jest-fixed-jsdom swaps in Node's fetch stack so MSW can intercept it, but
// leaves jsdom's File in place -- and a jsdom File isn't a Node Blob, so
// Node's FormData rejects it ("parameter 2 is not of type 'Blob'"). Every
// upload test builds a File and appends it to a FormData, so File has to
// come from the same realm as FormData and Blob.
const FixedJSDOMEnvironment = require("jest-fixed-jsdom");

class AppJSDOMEnvironment extends FixedJSDOMEnvironment {
  constructor(...args) {
    super(...args);
    this.global.File = File;
  }
}

module.exports = AppJSDOMEnvironment;
