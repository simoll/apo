import sys
from matplotlib import pyplot as pl

logFile = sys.argv[1]

# - Round 200 (global_step=199, learning_rate=0.001) -
def parseRoundHeader(l):
  parts = l.split(' ')
  trainRound = int(parts[2]) # '' 'Round' '200'
  return trainRound

# Greedy (best)  0.8790  (matched 0.8550, longerDer 0.0020, shorterDer: 0.0030, betterScore: 0.0190)
def parseEvalLine(l):
  # "best)  0.8790  "
  seg = l.split('(')
  leftParts = seg[1].split(' ')
  totalText = leftParts[-3] 
  total = float(totalText)
  return total

def parseLog(logPath):
  X = []
  bestY = []
  trainRound=None
  for l in open(logPath, 'r'):
    if l.startswith("- Round "):
      trainRound = parseRoundHeader(l)
    elif "(best)" in l:
      if trainRound is None:
        print("Missed round header!!!")
        raise SystemExit
      X.append(trainRound)
      # total, match, longerDer, shortedDer, betterScore = parseEvalLine(l)
      total = parseEvalLine(l)
      bestY.append(total)
  return X, bestY


X, Y = parseLog(sys.argv[1])
pl.plot(X, Y)
pl.show()
