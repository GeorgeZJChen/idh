import React, { Component } from 'react'
import '../css/footer.css'

class Footer extends Component {

  render() {
    return (
      <div className='footer'>
        <hr/>
        <p className='footer-text'>Welcome to this site! There are more interesting projects presented.</p>
        <div className='footer-mp' onClick={()=>this.moreProjects()}>{"See more projects >"}</div>
        <h3>Contact:</h3>
        <p className='footer-text'>By email: <a className='footer-email' href="mailto:georgechenzj@outlook.com">georgechenzj@outlook.com</a></p>
      </div>
    )
  }
  moreProjects(){
    if(!this.props.parent.refs.header.blockShows){
      this.props.parent.refs.header.moreProjects()
    }
  }
}

export default Footer
