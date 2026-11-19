package app.earshield.app

import android.app.Application

class EarshieldApp : Application() {
    lateinit var controller: ShieldController
        private set

    override fun onCreate() {
        super.onCreate()
        controller = ShieldController(this)
    }
}

val android.content.Context.controller: ShieldController
    get() = (applicationContext as EarshieldApp).controller
