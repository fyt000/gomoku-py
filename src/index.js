import React from 'react';
import ReactDOM from 'react-dom';
import './index.css';

function Square(props) {
  let valClass = "";
  if (props.value === 1) {
    valClass = 'X';
  }
  if (props.value === 2) {
    valClass = 'O';
  }
  return (
    <div className={valClass + " circle"} onClick={props.onClick}>
    </div>
  );
}

function calculateWinner(squares) {
  return fetch('http://localhost:5000/api/iswinner/', {
    method: 'post',
    body: JSON.stringify({
      board: squares
    }),
    mode: 'cors',
    headers: new Headers({
      'Content-Type': 'application/json'
    })
  })
    .then((response) => response.json())
    .then((responseJson) => {
      return responseJson.winner;
    })
    .catch((error) => {
      console.error(error);
    });
}

function makeNextMove(squares) {
   return fetch('http://localhost:5000/api/getnextmove/', {
    method: 'post',
    body: JSON.stringify({
      board: squares,
      cur: 2
    }),
    mode: 'cors',
    headers: new Headers({
      'Content-Type': 'application/json'
    })
  })
  .then((response) => {
    return response.json()
  })
  .catch((error) => {
      console.error(error);
  });
}

class Board extends React.Component {
  // constructor() {
  //   super();
  // }

  renderSquare(i, j) {
    return (<Square value={this.props.squares[i * 15 + j]}
      onClick={() => this.props.onClick(i, j)}
    />);
  }

  render() {
    var boardSquare = []
    for (let i = 0; i < 15; i++) {
      var rowSquare = []
      for (let j = 0; j < 15; j++) {
        rowSquare.push(this.renderSquare(i, j));
      }
      boardSquare.push(
        <div className="board-row">
          {rowSquare}
        </div>);
    }
    return (
      <div>
        {boardSquare}
      </div>
    );
  }
}

class Game extends React.Component {
  constructor() {
    super();
    this.state = {
      history: [{
        squares: Array(15 * 15).fill(0),
      }],
      stepNumber: 0,
      xIsNext: true,
      blocked: false,
      winner: 0,
    };
  }

  handleClick(i, j) {
    const history = this.state.history.slice(0, this.state.stepNumber + 1);
    const current = history[history.length - 1];
    const squares = current.squares.slice();
    if (squares[i * 15 + j] !== 0 || this.state.blocked || this.state.winner !== 0) {
      return;
    }
    squares[i * 15 + j] = this.state.xIsNext ? 1 : 2;
    this.setState({
      history: history.concat([{
        squares: squares
      }]),
      stepNumber: history.length,
      xIsNext: !this.state.xIsNext,
      blocked: true
    });

    calculateWinner(squares).then((theWinner) => {
      this.setState({
        winner: theWinner
      });
    })

    makeNextMove(squares).then((responseJson) => {
      const history = this.state.history.slice(0, this.state.stepNumber + 1);
      const current = history[history.length - 1];
      const squares = current.squares.slice();
      squares[responseJson.x * 15 + responseJson.y] = this.state.xIsNext ? 1 : 2;
      this.setState({
        history: history.concat([{
          squares: squares
        }]),
        stepNumber: history.length,
        xIsNext: !this.state.xIsNext,
        blocked: false
      });
    })
  }

  retract() {
    if (this.stepNumber !== 0) {
      this.setState({
        stepNumber: this.state.stepNumber - 2,
        xIsNext: ((this.state.stepNumber - 2) % 2) ? false : true,
      });
    }
  }

  restart() {
    this.setState({
      history: [{
        squares: Array(15 * 15).fill(0),
      }],
      stepNumber: 0,
      xIsNext: true,
      blocked: false,
      winner: 0,
    });
  }

  render() {
    const history = this.state.history;
    const current = history[this.state.stepNumber];

    let status;
    if (this.state.winner !== 0) {
      status = 'Winner: player ' + (this.state.winner);
    } else {
      status = 'Next player: ' + (this.state.xIsNext ? 'Black' : 'White');
    }
    let wait;
    if (this.state.blocked && this.state.winner === 0) {
      wait = "Waiting....";
    }

    return (
      <div className="game">
        <div className="game-board">
          <Board
            squares={current.squares}
            onClick={(i, j) => this.handleClick(i, j)}
          />
        </div>
        <div className="game-info">
          <div>{status}</div>
          <button onClick={() => this.retract()}>Retract</button>
          <button onClick={() => this.restart()}>Restart</button>
          <div>{wait}</div>
        </div>
      </div>
    );
  }
}

// ========================================

ReactDOM.render(
  <Game />,
  document.getElementById('root')
);
