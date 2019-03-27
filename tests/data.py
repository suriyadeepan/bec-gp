def test_load():
  from neuralbec import data
  import pickle
  # generate and save synthetic data
  datadict = { 'x' : [1, 2, 3],
      'data' : [
        (0.2, [ 0.2, 0.4, 0.6 ]),
        (0.3, [ 0.3, 0.6, 0.9 ]),
        (0.4, [ 0.4, 0.8, 1.2 ])
        ]
      }
  pickle.dump(datadict, open('data/dummy.data', 'wb'))
  inputs, outputs, reference = data.load('dummy')
  assert inputs == [ 0.2, 0.3, 0.4 ]
  assert outputs[-1] == datadict['data'][-1][-1]
  assert reference['x'] == datadict['x']
