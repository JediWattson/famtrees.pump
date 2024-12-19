const protobuf = require("protobufjs")
const { makeBuffer } = require('./helpers') 

const upsert = (data) => {
	protobuf.load("../records.proto", (err, root) => {
		if (err)
			throw err

		if (data.tokenizedLogs)
			return makeBuffer(root.lookupType('LogData'), data)
		
		makeBuffer(root.lookupType('BondingCurveData'), data)
	})
}

module.exports = { upsert }


