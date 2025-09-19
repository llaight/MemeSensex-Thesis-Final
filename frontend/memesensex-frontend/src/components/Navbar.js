import logo_ribbon from "../asset/logo_ribbon.svg";

const Navbar = () => {
    return ( 
        <nav className="bg-gradient-to-r from-purple-600 via-blue-600 to-teal-500 text-white py-4 px-6 shadow-lg">
            <div className="max-w-6xl mx-auto flex items-center gap-3">
                <div className="w-10 h-10 bg-white/100 rounded-full flex items-center justify-center">
                    <img src={logo_ribbon} alt="Logo" className="h-6 w-6 text-white"/>
                </div>
                <h1 className="text-2xl font-bold text-white">
                    MemeSenseX
                </h1>
            </div>
        </nav>
    );
}
 
export default Navbar;