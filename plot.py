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
  parts = l.split("matched")
  leftParts = parts[0][:-1].strip().split(" ")
  totalText = leftParts[-1] 

  detailParts = parts[1][:-1].split(' ')
  betterText = detailParts[-1][:-1]

  return float(totalText), float(betterText)

def parseLog(logPath):
  taskName=None
  X = []
  bestY = []
  bestBetterY = []
  stopY = []
  stopBetterY = []

  incBetterX = [] # only track improvements for incumbent
  incBetterY = []

  lastIncumbent=0.0
  trainRound=None
  for l in open(logPath, 'r'):
    if l.startswith("- Round "):
      trainRound = parseRoundHeader(l)
      X.append(trainRound)
    elif "(best)" in l:
      if trainRound is None:
        print("Missed round header!!!")
        raise SystemExit
      # total, match, longerDer, shortedDer, betterScore = parseEvalLine(l)
      bestTotal, bestBetter = parseEvalLine(l)
      bestY.append(bestTotal)
      bestBetterY.append(bestBetter)
    elif "(STOP)" in l:
      if trainRound is None:
        print("Missed round header!!!")
        raise SystemExit
      # total, match, longerDer, shortedDer, betterScore = parseEvalLine(l)
      stopTotal, stopBetter = parseEvalLine(l)
      stopY.append(stopTotal)
      stopBetterY.append(stopBetter)
    elif "Incumbent" in l:
      if trainRound is None:
        print("Missed round header!!!")
        raise SystemExit
      # total, match, longerDer, shortedDer, betterScore = parseEvalLine(l)

      incTotal, incBetter = parseEvalLine(l)
      if incBetter <= lastIncumbent:
        continue
        
      lastIncumbent=incBetter
      incBetterX.append(trainRound)
      incBetterY.append(incBetter)
    elif l.startswith("name ="):
      taskName = l.split(" ")[2].strip()

  return taskName, X, bestY, bestBetterY, stopY, stopBetterY, incBetterX, incBetterY

# parse data
taskName, X, bestY, bestBetterY, stopY, stopBetterY, incBetterX, incBetterY = parseLog(sys.argv[1])

# build an artificial performance score
def buildPerfIndex(bestY, bestBetterY):
  return [best + better for best, better in zip(bestY, bestBetterY)]

indexY = buildPerfIndex(bestY, bestBetterY)


# plot data
pl.plot(X, bestY, 'b', label="best")
pl.plot(X, bestBetterY, 'c', label="improved (best)")
pl.plot(X, stopY, 'g',label="stop")
pl.plot(X, stopBetterY, 'm',label="improved (stop)")
pl.plot(incBetterX, incBetterY, 'k.:',label="improved (ceil)")
if len(incBetterY) > 0:
  # append line w/o dot
  lastValue = incBetterY[-1]
  lastUpdateRound = incBetterX[-1]
  lastRound = X[-1]
  if lastUpdateRound != lastRound:
    pl.plot([lastUpdateRound, lastRound], [lastValue] * 2, 'k:')

pl.plot(X, indexY, 'k:',label="perf index")

pl.title("training on {} task".format(taskName))

# axis
pl.xlabel("global step")
pl.ylabel("% of hold-out samples")
pl.legend()
pl.show()
